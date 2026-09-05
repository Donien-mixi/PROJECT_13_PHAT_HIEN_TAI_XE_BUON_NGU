#ifndef PNP_SOLVER_H_
#define PNP_SOLVER_H_

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float yaw;    // Rotation around vertical Y-axis (degrees): Left (+) / Right (-)
    float pitch;  // Rotation around horizontal X-axis (degrees): Up (+) / Down (-)
    float roll;   // Rotation around optical Z-axis (degrees): Tilt
    float tx;     // Estimated translation X (mm)
    float ty;     // Estimated translation Y (mm)
    float tz;     // Estimated distance Z (mm)
    bool is_valid;
} head_pose_t;

typedef struct {
    float x; // Normalized [0.0, 1.0]
    float y; // Normalized [0.0, 1.0]
} point2d_t;

/**
 * Initializes the 3D anthropometric face model.
 */
void pnp_solver_init(void);

/**
 * Solves 3D Head Pose (Yaw, Pitch, Roll) using pure C++ POSIT/PnP
 * without requiring OpenCV or external linear algebra libraries.
 *
 * @param landmarks_22 Array of 22 normalized facial keypoints [0.0, 1.0].
 * @param out_pose Pointer to store the calculated head pose angles.
 * @return true if convergence reached, false on degenerate input.
 */
bool pnp_solve_head_pose(const point2d_t landmarks_22[22], head_pose_t* out_pose);

#ifdef __cplusplus
}
#endif

#endif // PNP_SOLVER_H_
