
extern crate bindgen;
extern crate cc;

use std::env;
use std::path::PathBuf;

fn main() {
    // println!("cargo:rerun-if-changed=../../rabitqlib");
    // println!("cargo:rerun-if-changed=../../rabitqlib/**/*.hpp");
    // println!("cargo:rerun-if-changed=../../rabitqlib/**/*.cpp");
    // println!("cargo:rerun-if-changed=rabitq_wrapper.cpp");
    // println!("cargo:rerun-if-changed=rabitq.h");

    // 递归检测rabitqlib目录下的所有头文件和源文件
    let rabitqlib_path = "../../rabitqlib";
    if let Ok(entries) = std::fs::read_dir(rabitqlib_path) {
        for entry in entries.flatten() {
            if let Ok(file_type) = entry.file_type() {
                if file_type.is_file() {
                    if let Some(ext) = entry.path().extension() {
                        if ext == "hpp" || ext == "cpp" {
                            println!("cargo:rerun-if-changed={}", entry.path().display());
                        }
                    }
                }
            }
        }
    }
    
    // Compile the C++ wrapper.
    cc::Build::new()
        .file("rabitq_wrapper.cpp")
        .include("../../rabitqlib") 
        .cpp(true)
        .flag("-std=c++17")
        .flag("-march=native")
        .flag("-mavx512f")
        .flag("-fopenmp")
        .compile("rabitq_wrapper");

    println!("cargo:rustc-link-lib=stdc++");

    // Generate bindings for the C header.
    let bindings = bindgen::Builder::default()
        .header("rabitq.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}