#pragma once
// =============================================================================
// Geometry.hpp — rigid bodies immersed in the flow (ported from
// panchiittp/meshfree4bgk2d src/Geometry.hpp, de-templated; 2D only).
//
// Static bodies for now: surface particles are generated once, carry the
// body's normal / wall velocity / temperature, and participate in the
// diffuse-reflection BC exactly like box-wall particles. The dynamic state
// (velocity, force, torque) is kept so a moving-body update can be added
// on top without changing the interface.
// =============================================================================
#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

struct RigidBody
{
    int    Npts;              // surface particles to generate
    double cx, cy;            // centre of mass
    double cz = 0.0;          // centre z (3-D bodies)
    double mass;
    double T;                 // surface temperature
    double ox, oy;            // orientation unit vector (2-D bodies)
    double velx = 0.0, vely = 0.0, velz = 0.0;
    double angularVel = 0.0;                       // 2-D scalar spin
    double omx = 0.0, omy = 0.0, omz = 0.0;        // 3-D angular velocity
    bool   rotates = true;   // free to spin under the hydrodynamic torque
    double forcex = 0.0, forcey = 0.0, forcez = 0.0;
    double torque = 0.0;                           // 2-D scalar torque
    double tqx = 0.0, tqy = 0.0, tqz = 0.0;        // 3-D torque vector
    int    dim = 2;

    RigidBody(int Npts_, double cx_, double cy_, double mass_, double T_,
              double ox_ = 1.0, double oy_ = 0.0)
        : Npts(Npts_), cx(cx_), cy(cy_), mass(mass_), T(T_), ox(ox_), oy(oy_) {}

    virtual ~RigidBody() = default;

    // Orientation access for checkpointing. Only the 3-D cube carries a
    // rotation matrix; everything else returns nullptr and restores from the
    // (ox, oy) pair alone, which is all its shape needs.
    virtual const double *orientation() const { return nullptr; }
    virtual void setOrientation(const double *) {}

    /* True if (x, y) lies inside the body. */
    virtual bool isInObject(double x, double y) const = 0;

    /* Append Npts surface-particle positions. */
    virtual void generateBoundaryParticles(
        std::vector<std::pair<double, double>> &vec) const = 0;

    /* Outward normal (into the gas) at surface point (x, y). */
    virtual void getNormal(double x, double y,
                           double *nx, double *ny) const = 0;

    /* Gas-facing surface length associated with one surface particle. */
    virtual double getSurfaceArea() const = 0;

    virtual double getMomentOfInertia() const = 0;

    /* ── 3-D interface (implemented by 3-D bodies such as Sphere) ────── */
    virtual bool isInObject3(double x, double y, double z) const
    { (void)z; return isInObject(x, y); }

    virtual void generateBoundaryParticles3(
        std::vector<std::array<double, 3>> &vec) const { (void)vec; }

    virtual void getNormal3(double x, double y, double z,
                            double *nx, double *ny, double *nz) const
    { (void)z; getNormal(x, y, nx, ny); *nz = 0.0; }

    /* Advance a 3-D body's orientation by the rotation w for time dt.
       Bodies whose surface state is orientation-free (Sphere) ignore it. */
    virtual void rotate3(double wx, double wy, double wz, double dt)
    { (void)wx; (void)wy; (void)wz; (void)dt; }
};

/* Sphere — genuine 3-D body. Near-uniform surface sampling via a
   Fibonacci lattice; radial normals; isotropic inertia I = (2/5) m R²,
   so the 3-D Newton-Euler update is dω/dt = τ / I componentwise. */
struct Sphere : public RigidBody
{
    double R;   // radius

    Sphere(double cx_, double cy_, double cz_, double density, double T_,
           double R_, int Npts_)
        : RigidBody(Npts_, cx_, cy_, density * 4.0 / 3.0 * M_PI * R_ * R_ * R_,
                    T_),
          R(R_)
    {
        cz = cz_;
        dim = 3;
    }

    double getMomentOfInertia() const override { return 0.4 * mass * R * R; }
    double getSurfaceArea() const override
    { return 4.0 * M_PI * R * R / Npts; }

    bool isInObject(double x, double y) const override
    { return isInObject3(x, y, cz); }

    bool isInObject3(double x, double y, double z) const override
    {
        double dx = x - cx, dy = y - cy, dz = z - cz;
        return dx * dx + dy * dy + dz * dz <= R * R;
    }

    void getNormal(double x, double y, double *nx, double *ny) const override
    {
        double nz;
        getNormal3(x, y, cz, nx, ny, &nz);
    }

    void getNormal3(double x, double y, double z,
                    double *nx, double *ny, double *nz) const override
    {
        double dx = x - cx, dy = y - cy, dz = z - cz;
        double n = std::sqrt(dx * dx + dy * dy + dz * dz);
        *nx = dx / n; *ny = dy / n; *nz = dz / n;
    }

    void generateBoundaryParticles(
        std::vector<std::pair<double, double>> &vec) const override
    { (void)vec; }   // 3-D body: use generateBoundaryParticles3

    void generateBoundaryParticles3(
        std::vector<std::array<double, 3>> &vec) const override
    {
        // Fibonacci sphere: i-th point at golden-angle longitude
        const double ga = M_PI * (3.0 - std::sqrt(5.0));
        for (int i = 0; i < Npts; ++i)
        {
            double zf  = 1.0 - 2.0 * (i + 0.5) / Npts;   // (-1, 1)
            double rf  = std::sqrt(1.0 - zf * zf);
            double th  = ga * i;
            vec.push_back({cx + R * rf * std::cos(th),
                           cy + R * rf * std::sin(th),
                           cz + R * zf});
        }
    }
};

struct Square : public RigidBody
{
    double L;   // side length

    Square(double cx_, double cy_, double density, double T_, double L_,
           int Npts_, double ox_ = 0.0, double oy_ = 1.0)
        : RigidBody(Npts_, cx_, cy_, density * L_ * L_, T_, ox_, oy_), L(L_)
    {
        double n = std::sqrt(ox * ox + oy * oy);
        ox /= n; oy /= n;
    }

    double getMomentOfInertia() const override { return mass * L * L / 6.0; }
    double getSurfaceArea() const override { return 4.0 * L / Npts; }

    void getNormal(double x, double y, double *nx, double *ny) const override
    {
        double xp = x - cx, yp = y - cy;
        double n = std::sqrt(xp * xp + yp * yp);
        // Side whose outward normal has the smallest angle to (xp, yp)
        double dots[4][3] = {
            { (xp * ox + yp * oy) / n,  ox,  oy},   // top
            {-(xp * ox + yp * oy) / n, -ox, -oy},   // bottom
            { (xp * oy - yp * ox) / n,  oy, -ox},   // right
            {-(xp * oy - yp * ox) / n, -oy,  ox},   // left
        };
        int best = 0;
        for (int i = 1; i < 4; ++i)
            if (dots[i][0] > dots[best][0]) best = i;
        *nx = dots[best][1];
        *ny = dots[best][2];
    }

    bool isInObject(double x, double y) const override
    {
        double xC = x - cx, yC = y - cy;
        double D  = ox * ox + oy * oy;
        double xp = (ox * xC + oy * yC) / D;
        double yp = (oy * xC - ox * yC) / D;
        return (std::fabs(xp) <= L / 2) && (std::fabs(yp) <= L / 2);
    }

    void generateBoundaryParticles(
        std::vector<std::pair<double, double>> &vec) const override
    {
        double t = std::atan2(oy, ox);
        double c = std::cos(t), s = std::sin(t);
        int    nSide = Npts / 4;
        double dl = L / nSide;

        double topLeftX = cx + L * (c - s) / 2, topLeftY = cy + L * (s + c) / 2;
        double botRightX = cx + L * (s - c) / 2, botRightY = cy - L * (s + c) / 2;

        for (int i = 0; i < nSide; ++i)   // right side
            vec.emplace_back(botRightX + dl * (i + 0.5) * c,
                             botRightY + dl * (i + 0.5) * s);
        for (int i = 0; i < nSide; ++i)   // left side
            vec.emplace_back(topLeftX - dl * (i + 0.5) * c,
                             topLeftY - dl * (i + 0.5) * s);
        for (int i = 0; i < nSide; ++i)   // bottom side
            vec.emplace_back(botRightX - dl * (i + 0.5) * s,
                             botRightY + dl * (i + 0.5) * c);
        for (int i = 0; i < nSide; ++i)   // top side
            vec.emplace_back(topLeftX + dl * (i + 0.5) * s,
                             topLeftY - dl * (i + 0.5) * c);
    }
};

struct Circle : public RigidBody
{
    double R;   // radius

    Circle(double cx_, double cy_, double density, double T_, double R_,
           int Npts_)
        : RigidBody(Npts_, cx_, cy_, density * M_PI * R_ * R_, T_), R(R_) {}

    double getMomentOfInertia() const override { return mass * R * R / 2.0; }
    double getSurfaceArea() const override { return 2.0 * M_PI * R / Npts; }

    void getNormal(double x, double y, double *nx, double *ny) const override
    {
        double dx = x - cx, dy = y - cy;
        double n = std::sqrt(dx * dx + dy * dy);
        *nx = dx / n;
        *ny = dy / n;
    }

    bool isInObject(double x, double y) const override
    {
        double dx = x - cx, dy = y - cy;
        return dx * dx + dy * dy <= R * R;
    }

    void generateBoundaryParticles(
        std::vector<std::pair<double, double>> &vec) const override
    {
        double dth = 2.0 * M_PI / Npts;
        for (int i = 0; i < Npts; ++i)
            vec.emplace_back(R * std::cos(i * dth) + cx,
                             R * std::sin(i * dth) + cy);
    }
};

/* Cube — oriented 3-D body. Orientation carried as a body-to-world
   rotation matrix Rm, advanced each step by the Rodrigues increment of
   w dt (rotate3). The inertia tensor of a cube about its centre is
   isotropic (I = m L^2 / 6 per axis), so the vector Newton-Euler update
   with a scalar I is exact. Surface: n x n particles per face. */
struct Cube : public RigidBody
{
    double L;                                   // side length
    double Rm[9] = {1,0,0, 0,1,0, 0,0,1};       // body -> world
    const double *orientation() const override { return Rm; }
    void setOrientation(const double *m) override
    {
        for (int k = 0; k < 9; ++k) Rm[k] = m[k];
    }

    Cube(double cx_, double cy_, double cz_, double density, double T_,
         double L_, int Npts_)
        : RigidBody(Npts_, cx_, cy_, density * L_ * L_ * L_, T_), L(L_)
    {
        cz = cz_;
        dim = 3;
    }

    double getMomentOfInertia() const override { return mass * L * L / 6.0; }
    double getSurfaceArea() const override { return 6.0 * L * L / Npts; }

    void toBody(double x, double y, double z,
                double *bx, double *by, double *bz) const
    {
        double dx = x - cx, dy = y - cy, dz = z - cz;
        *bx = Rm[0]*dx + Rm[3]*dy + Rm[6]*dz;   // R^T (world -> body)
        *by = Rm[1]*dx + Rm[4]*dy + Rm[7]*dz;
        *bz = Rm[2]*dx + Rm[5]*dy + Rm[8]*dz;
    }

    bool isInObject(double x, double y) const override
    { return isInObject3(x, y, cz); }

    bool isInObject3(double x, double y, double z) const override
    {
        double bx, by, bz;
        toBody(x, y, z, &bx, &by, &bz);
        return std::fabs(bx) <= L/2 && std::fabs(by) <= L/2 &&
               std::fabs(bz) <= L/2;
    }

    void getNormal(double x, double y, double *nx, double *ny) const override
    { double nz; getNormal3(x, y, cz, nx, ny, &nz); }

    void getNormal3(double x, double y, double z,
                    double *nx, double *ny, double *nz) const override
    {
        double b[3]; toBody(x, y, z, &b[0], &b[1], &b[2]);
        int f = 0;
        for (int i = 1; i < 3; ++i)
            if (std::fabs(b[i]) > std::fabs(b[f])) f = i;
        double s = (b[f] >= 0.0) ? 1.0 : -1.0;   // face normal in body frame
        *nx = s * Rm[0 + f];                      // column f of Rm
        *ny = s * Rm[3 + f];
        *nz = s * Rm[6 + f];
    }

    void generateBoundaryParticles(
        std::vector<std::pair<double, double>> &vec) const override
    { (void)vec; }   // 3-D body: use generateBoundaryParticles3

    void generateBoundaryParticles3(
        std::vector<std::array<double, 3>> &vec) const override
    {
        int n = std::max(1, (int)std::lround(std::sqrt(Npts / 6.0)));
        double dl = L / n;
        for (int f = 0; f < 3; ++f)          // face-normal axis
            for (int sgn = -1; sgn <= 1; sgn += 2)
                for (int i = 0; i < n; ++i)
                    for (int j = 0; j < n; ++j)
                    {
                        double b[3];
                        b[f] = sgn * L / 2.0;
                        b[(f+1)%3] = (i + 0.5) * dl - L / 2.0;
                        b[(f+2)%3] = (j + 0.5) * dl - L / 2.0;
                        vec.push_back({cx + Rm[0]*b[0] + Rm[1]*b[1] + Rm[2]*b[2],
                                       cy + Rm[3]*b[0] + Rm[4]*b[1] + Rm[5]*b[2],
                                       cz + Rm[6]*b[0] + Rm[7]*b[1] + Rm[8]*b[2]});
                    }
    }

    void rotate3(double wx, double wy, double wz, double dt) override
    {
        double w = std::sqrt(wx*wx + wy*wy + wz*wz);
        if (w < 1e-300) return;
        double ax = wx/w, ay = wy/w, az = wz/w;
        double th = w * dt, c = std::cos(th), s = std::sin(th), C = 1.0 - c;
        double D[9] = { c + ax*ax*C,    ax*ay*C - az*s, ax*az*C + ay*s,
                        ay*ax*C + az*s, c + ay*ay*C,    ay*az*C - ax*s,
                        az*ax*C - ay*s, az*ay*C + ax*s, c + az*az*C };
        double Rn[9];
        for (int r = 0; r < 3; ++r)
            for (int col = 0; col < 3; ++col)
                Rn[3*r+col] = D[3*r+0]*Rm[0+col] + D[3*r+1]*Rm[3+col]
                            + D[3*r+2]*Rm[6+col];
        // Re-orthonormalise (Gram-Schmidt on rows) against drift
        auto nrm = [&](int r){ double m = std::sqrt(Rn[3*r]*Rn[3*r]
                       + Rn[3*r+1]*Rn[3*r+1] + Rn[3*r+2]*Rn[3*r+2]);
                       for (int k = 0; k < 3; ++k) Rn[3*r+k] /= m; };
        nrm(0);
        double d = Rn[0]*Rn[3] + Rn[1]*Rn[4] + Rn[2]*Rn[5];
        for (int k = 0; k < 3; ++k) Rn[3+k] -= d * Rn[k];
        nrm(1);
        Rn[6] = Rn[1]*Rn[5] - Rn[2]*Rn[4];
        Rn[7] = Rn[2]*Rn[3] - Rn[0]*Rn[5];
        Rn[8] = Rn[0]*Rn[4] - Rn[1]*Rn[3];
        for (int k = 0; k < 9; ++k) Rm[k] = Rn[k];
    }
};
