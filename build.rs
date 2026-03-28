fn main() {
    cc::Build::new()
        .file("csrc/math_ops.c")
        .include("csrc")
        .opt_level(2)
        .warnings(true)
        .compile("stenowav_math");

    println!("cargo:rerun-if-changed=csrc/math_ops.c");
    println!("cargo:rerun-if-changed=csrc/math_ops.h");
}
