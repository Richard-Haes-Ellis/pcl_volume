#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>
#include <pcl/filters/filter.h>  // for removeNaNFromPointCloud
#include <pcl/common/common.h>   // for isFinite
#include <pcl/io/vtk_io.h>

int main ()
{
  // Load input file into a PointCloud<T> with an appropriate type
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PCLPointCloud2 cloud_blob;

  // Load the PCD file
  pcl::io::loadPCDFile ("../pcd/saved_pointcloud.pcd", cloud_blob);
  pcl::fromPCLPointCloud2 (cloud_blob, *cloud);

  // Remove NaNs from the cloud
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*cloud, *cloud, indices);

  // Check if the cloud is empty after removing NaNs
  if (cloud->points.empty()) {
    std::cerr << "Error: Point cloud is empty after removing NaNs!" << std::endl;
    return (-1);
  }

  // Remove points with infinite values
  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud (new pcl::PointCloud<pcl::PointXYZ>);
  for (const auto& point : cloud->points) {
    if (pcl::isFinite(point)) {
      filtered_cloud->points.push_back(point);
    } else {
      std::cerr << "Invalid point detected and removed: (" 
                << point.x << ", " << point.y << ", " << point.z << ")" << std::endl;
    }
  }

  // Update cloud with filtered_cloud (without infinite points)
  filtered_cloud->width = static_cast<uint32_t>(filtered_cloud->points.size());
  filtered_cloud->height = 1;
  filtered_cloud->is_dense = true;

  // Ensure the cloud is not empty after filtering infinite points
  if (filtered_cloud->points.empty()) {
    std::cerr << "Error: Point cloud is empty after removing infinite values!" << std::endl;
    return (-1);
  }

  // Proceed with normal estimation and KdTree setup as before
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;
  pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);

  tree->setInputCloud(filtered_cloud);
  n.setInputCloud(filtered_cloud);
  n.setSearchMethod(tree);
  n.setKSearch(20);
  n.compute(*normals);

  // Concatenate the XYZ and normal fields
  pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::PointNormal>);
  pcl::concatenateFields(*filtered_cloud, *normals, *cloud_with_normals);

  pcl::search::KdTree<pcl::PointNormal>::Ptr tree2 (new pcl::search::KdTree<pcl::PointNormal>);
  tree2->setInputCloud(cloud_with_normals);

  pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;
  pcl::PolygonMesh triangles;

  gp3.setSearchRadius(0.025);
  gp3.setMu(2.5);
  gp3.setMaximumNearestNeighbors(100);
  gp3.setMaximumSurfaceAngle(M_PI / 4);
  gp3.setMinimumAngle(M_PI / 18);
  gp3.setMaximumAngle(2 * M_PI / 3);
  gp3.setNormalConsistency(false);

  gp3.setInputCloud(cloud_with_normals);
  gp3.setSearchMethod(tree2);
  gp3.reconstruct(triangles);

  // Save the result to disk
    pcl::io::saveVTKFile ("mesh.vtk", triangles);


  return (0);
}
