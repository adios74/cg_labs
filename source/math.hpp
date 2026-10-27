#pragma once

#include <cmath>
#include <cstring>

namespace math {

using Mat4 = float[4][4];

inline void identity(Mat4 out) {
    std::memset(out, 0, sizeof(Mat4));
    out[0][0] = out[1][1] = out[2][2] = out[3][3] = 1.0f;
}

inline void multiply(const Mat4 a, const Mat4 b, Mat4 out) {
    Mat4 result{};
    for (int column = 0; column < 4; ++column) {
        for (int row = 0; row < 4; ++row) {
            float sum = 0.0f;
            for (int k = 0; k < 4; ++k) {
                sum += a[k][row] * b[column][k];
            }
            result[column][row] = sum;
        }
    }
    std::memcpy(out, result, sizeof(Mat4));
}

inline void translate(float x, float y, float z, Mat4 out) {
    identity(out);
    out[3][0] = x;
    out[3][1] = y;
    out[3][2] = z;
}

inline void scale(float x, float y, float z, Mat4 out) {
    std::memset(out, 0, sizeof(Mat4));
    out[0][0] = x;
    out[1][1] = y;
    out[2][2] = z;
    out[3][3] = 1.0f;
}

inline void rotateX(float angle, Mat4 out) {
    identity(out);
    const float c = std::cos(angle);
    const float s = std::sin(angle);
    out[1][1] = c;
    out[1][2] = s;
    out[2][1] = -s;
    out[2][2] = c;
}

inline void rotateY(float angle, Mat4 out) {
    identity(out);
    const float c = std::cos(angle);
    const float s = std::sin(angle);
    out[0][0] = c;
    out[0][2] = -s;
    out[2][0] = s;
    out[2][2] = c;
}

inline void rotateZ(float angle, Mat4 out) {
    identity(out);
    const float c = std::cos(angle);
    const float s = std::sin(angle);
    out[0][0] = c;
    out[0][1] = s;
    out[1][0] = -s;
    out[1][1] = c;
}

inline void perspective(float fov_y, float aspect, float near_z, float far_z, Mat4 out) {
    std::memset(out, 0, sizeof(Mat4));
    const float f = 1.0f / std::tan(fov_y * 0.5f);
    out[0][0] = f / aspect;
    out[1][1] = f;
    out[2][2] = far_z / (near_z - far_z);
    out[2][3] = -1.0f;
    out[3][2] = near_z * far_z / (near_z - far_z);
}

inline void ortho(float left, float right, float bottom, float top,
                  float near_z, float far_z, Mat4 out) {
    std::memset(out, 0, sizeof(Mat4));
    out[0][0] = 2.0f / (right - left);
    out[1][1] = 2.0f / (top - bottom);
    out[2][2] = 1.0f / (near_z - far_z);
    out[3][0] = -(right + left) / (right - left);
    out[3][1] = -(top + bottom) / (top - bottom);
    out[3][2] = near_z / (near_z - far_z);
    out[3][3] = 1.0f;
}

}
