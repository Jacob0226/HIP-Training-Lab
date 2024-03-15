#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/scan.h>
#include <hip/hip_runtime.h>
#include <iostream>
#include <string>
#include <fstream>

using namespace std;

void CHECK(const char* msg){
	hipError_t e = hipGetLastError();
	if(e!=hipSuccess){
		printf("hip error: %s\n", hipGetErrorString(e));
		printf("  Comment: %s\n", msg);
		exit(0);
	}
}

__host__ __device__ float3 operator+(const float3& a, const float3& b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__host__ __device__ float3 operator-(const float3& a, const float3& b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

std::ostream& operator<<(std::ostream& os, const float3& a) {
    os << "(" << a.x << ", " << a.y << ", " << a.z << ")\n";
    return os;
}

std::ostream& operator<<(std::ostream& os, const int3& a) {
    os << "(" << a.x << ", " << a.y << ", " << a.z << ")\n";
    return os;
}

struct PC_info{
	float3 min_coord;
	float3 max_coord;
	int3 gridSize;

	float r; // voxel size
	int3 voxel_dim; // how many voxels in x, y, and z dimension
};


void LoadPC(string pcd_path, vector<float3>& vertex, PC_info& info){
	std::ifstream file(pcd_path);
	if (!file.is_open()) {
        printf("Cannot open the file %s.\n", pcd_path.c_str());
        exit(0);
    }

    float3 v;
    while (file >> v.x >> v.y >> v.z) {
		info.min_coord.x = min(info.min_coord.x, v.x);
		info.min_coord.y = min(info.min_coord.x, v.y);
		info.min_coord.z = min(info.min_coord.x, v.z);
		info.max_coord.x = max(info.max_coord.x, v.x);
		info.max_coord.y = max(info.max_coord.x, v.y);
		info.max_coord.z = max(info.max_coord.x, v.z);
        vertex.push_back(v);
    }
}

void SavePC(vector<float3> vertex, float radius){
	string filename = "Result/downsampled_" + to_string(radius) + ".ply";
	std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return;
    }

	outfile << "ply\n";
    outfile << "format ascii 1.0\n";
    outfile << "element vertex " << vertex.size() << "\n";
    outfile << "property float x\n";
    outfile << "property float y\n";
    outfile << "property float z\n";
    outfile << "end_header\n";

    for (const auto& elem : vertex) {
        outfile << elem.x << " " << elem.y << " " << elem.z << std::endl;
    }

    outfile.close();
}

// Use thrust API to do the voxel grid filter
__global__ void GetVoxelGridIdx(float3* vertex, PC_info info, int* voxel_idx,int n){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid>=n) return;
	
	float r = info.r;
	int3 voxel_dim = info.voxel_dim;
	float3 diff = vertex[tid] - info.min_coord;
	int3 idx_xyz = {int(diff.x/r), int(diff.y/r), int(diff.z/r)};
	int idx = idx_xyz.x + idx_xyz.y * voxel_dim.x + idx_xyz.z * voxel_dim.x * voxel_dim.y;
	voxel_idx[tid] = idx;
}

struct Float3Plus {
    __host__ __device__ float3 operator()(const float3& a, const float3& b) const {
        return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
    }
};

struct DivideFloat3ByInt {
    __host__ __device__ float3 operator()(const float3& a, const int& b) const {
        return make_float3(a.x / static_cast<float>(b),
                           a.y / static_cast<float>(b),
                           a.z / static_cast<float>(b));
    }
};


int divUp(int a, int b) { return (a+b-1)/b;}

void VoxelGrid(std::vector<float3>& h_vertex, PC_info& info, std::vector<float3>& h_result){
	// Malloc device memory
	printf("VoxelGrid starts\n");
	float3* d_vertex;
	int* d_voxel_idx;
	int num_points = h_vertex.size();
	hipMallocManaged(&d_vertex, sizeof(float3) * num_points);
	hipMallocManaged(&d_voxel_idx, sizeof(int) * num_points);
	hipMemcpy(d_vertex, h_vertex.data(), sizeof(float3) * num_points, hipMemcpyHostToDevice);
	hipDeviceSynchronize();
	CHECK("Error!! Malloc");

	// Init key(voxel grid index) and data(canonical model vertices).
	dim3 block(512, 1, 1); // copy vertex to points_d_query
	dim3 grid(divUp(h_vertex.size(), block.x), 1, 1);
	GetVoxelGridIdx<<<grid, block>>>(d_vertex, info, d_voxel_idx, num_points);
	hipDeviceSynchronize();
	CHECK("Error!! GetVoxelGridIdx");
	thrust::device_vector<int> d_key(d_voxel_idx, d_voxel_idx + num_points);
	thrust::device_vector<float3> d_data(d_vertex, d_vertex+num_points);

	// Step 1. Sort the keys
    thrust::sort_by_key(d_key.begin(), d_key.end(), d_data.begin());
	hipDeviceSynchronize();
	CHECK("Error!! Step 1.");

	// Step 2. Sum up the occurrence of the index
	thrust::device_vector<int> d_one(num_points);
	thrust::fill(d_one.begin(), d_one.end(), 1);
    thrust::device_vector<int> d_reduce_key(num_points);
    thrust::device_vector<float3> d_reduce_sum(num_points);
    thrust::device_vector<int> d_count(num_points);

    thrust::reduce_by_key(
        d_key.begin(), d_key.end(),
        d_one.begin(),
        d_reduce_key.begin(),
        d_count.begin(),
        thrust::equal_to<int>(),
        thrust::plus<int>()
    );
	hipDeviceSynchronize();
	CHECK("Error!! Step 2.");

	// Step 3. Sum up the f3 coordinate
    auto key_end = thrust::reduce_by_key(
        d_key.begin(), d_key.end(),
        d_data.begin(),
        d_reduce_key.begin(),
        d_reduce_sum.begin(),
        thrust::equal_to<int>()
    ).first;
	int n_distinct_keys = key_end - d_reduce_key.begin();
	d_reduce_key.resize(n_distinct_keys);
	d_reduce_sum.resize(n_distinct_keys);
	d_count.resize(n_distinct_keys);
	hipDeviceSynchronize();
	CHECK("Error!! Step 3.");

	// Step 4. Divide the summed coordinate by d_count
    thrust::device_vector<float3> d_centroid(n_distinct_keys);
    thrust::transform(d_reduce_sum.begin(), d_reduce_sum.end(), d_count.begin(), d_centroid.begin(), DivideFloat3ByInt());
    hipDeviceSynchronize();
	CHECK("Error!! Step 4.");

	// Step 5. Pick the sampled vertices
	float3* ptr = thrust::raw_pointer_cast(d_centroid.data());
	h_result.resize(n_distinct_keys);
	hipMemcpy(h_result.data(), ptr, sizeof(float3) * n_distinct_keys, hipMemcpyDeviceToHost);
	hipDeviceSynchronize();
	CHECK("Error!! Step 5.");
	printf("The downsample size = %d vertices\n", n_distinct_keys);
}

int main(int argc, char* argv[]){
	if(argc<2){
		printf("Please specify point cloud file. e.g., ./program /path/to/pcd.ply\n");
		exit(0);
	}

	// Load point cloud from a txt file
	string pcd_path(argv[1]); 
	vector<float3> vertex;
	PC_info info;
	LoadPC(pcd_path, vertex, info);

	// Downsampling the point cloud
	float radius[3] = {0.1, 0.2, 0.4};
	for(int i=0; i<3; ++i){
		info.r = radius[i];
		float3 diff = info.max_coord - info.min_coord;
		info.voxel_dim.x = ceil(diff.x/info.r);
		info.voxel_dim.y = ceil(diff.y/info.r);
		info.voxel_dim.z = ceil(diff.z/info.r);
		
		vector<float3>result;
		VoxelGrid(vertex, info, result);
		SavePC(result, radius[i]);
	}
}