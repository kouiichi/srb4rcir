# Third-party notices

## acados, BLASFEO, and HPIPM

The NMPC production backend uses the acados source tree pinned in
`third_party/acados/`.  The checkout was built with shared libraries and the
generic BLASFEO/HPIPM targets.  The exact source revision is recorded by:

```bash
git -C third_party/acados rev-parse HEAD
```

acados, BLASFEO, and HPIPM are distributed under the 2-Clause BSD License;
the corresponding license texts are retained in their source checkout.  The
generated solver links against `third_party/acados/lib/libacados.so`,
`libhpipm.so`, and `libblasfeo.so`.

The Python template interface is installed editable from
`third_party/acados/interfaces/acados_template`.  It is used with the Isaac
Sim Python interpreter selected by `srb_0.0.5_env.sh`.

## CasADi, OSQP, SciPy, and Deprecated

These are runtime dependencies of the Isaac Sim Python environment.  Their
license and version metadata remain owned by that environment; this project
does not copy their source into the repository.

