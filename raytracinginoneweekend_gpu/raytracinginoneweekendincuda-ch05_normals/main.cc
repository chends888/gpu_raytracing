//Adaptado de: https://github.com/rogerallen/raytracinginoneweekendincuda/tree/ch05_normals

#include <iostream>
#include <chrono>
#include <fstream>
#include "sphere.h"
#include "hitable_list.h"
#include "float.h"

vec3 color(const ray& r, hitable *world) {
    hit_record rec;
    if (world->hit(r, 0.0, MAXFLOAT, rec)) {
        return 0.5*vec3(rec.normal.x()+1, rec.normal.y()+1, rec.normal.z()+1);
    }
    else {
        vec3 unit_direction = unit_vector(r.direction());
        float t = 0.5*(unit_direction.y() + 1.0);
        return (1.0-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
    }
}

int main() {
    int nx;
    int ny;
    std::cout << "Insert number of x pixels: ";
    std::cin >> nx;
    std::cout << "Insert number of y pixels: ";
    std::cin >> ny;
    auto start = std::chrono::high_resolution_clock::now();

    std::ofstream myfile;
    myfile.open ("image.ppm");
    myfile << "P3\n" << nx << " " << ny << "\n255\n";

    vec3 lower_left_corner(-2.0, -1.0, -1.0);
    vec3 horizontal(4.0, 0.0, 0.0);
    vec3 vertial(0.0, 2.0, 0.0);
    vec3 origin(0.0, 0.0, 0.0);
    hitable *list[2];
    list[0] = new sphere(vec3(0,0,-1), 0.5);
    list[1] = new sphere(vec3(0,-1.3,-1), 1);
    hitable *world = new hitable_list(list,2);
    for (int j = ny-1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            float u = float(i) / float(nx);
            float v = float(j) / float(ny);
            ray r(origin, lower_left_corner + u*horizontal + v*vertial);
            vec3 col = color(r, world);
            int ir = int(255.99*col[0]);
            int ig = int(255.99*col[1]);
            int ib = int(255.99*col[2]);
            myfile << ir << " " << ig << " " << ib << "\n";
        }
    }
    myfile.close();
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "Elapsed time for " << nx << "x" << ny << " image: " << elapsed.count() << " s\n";
}
