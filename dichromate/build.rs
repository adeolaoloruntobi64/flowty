#[cfg(windows)]
fn main() {
    // Now I don't need to do
    // #[cfg(target_os = "windows")]
    // enigo::set_dpi_awareness().unwrap();
    use embed_manifest::{manifest::DpiAwareness, new_manifest, embed_manifest};
    embed_manifest(
        new_manifest("Contoso.Sample")
            .dpi_awareness(DpiAwareness::PerMonitorV2)
    ).expect("unable to embed manifest");
    println!("cargo:rerun-if-changed=build.rs");
}

#[cfg(not(windows))]
fn main() {
    
}