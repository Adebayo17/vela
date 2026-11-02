"""
sw_build.py — VELA Phase 3: Vitis platform, bare-metal application and BOOT.BIN

Run by the Vitis 2024.2 Python interpreter (make sw):
    vitis -s tools/sw_build.py <xsa> <workspace> <out_dir>

    1. platform component from the XSA of make fpga: standalone domain on
       ps7_cortexa9_0 with the FatFs library (xilffs) and long file names
       (test_batch.bin is not an 8.3 name); the FSBL is built with it
    2. application component vela_app from sw/vela_app/src (empty template),
       linked with libm (sqrt, exp)
    3. BOOT.BIN = FSBL + bitstream + vela_app.elf (bootgen), for an SD card
       boot: copy BOOT.BIN and test_batch.bin to the card (VELA-TN-007 §10.4)

Everything is regenerated: the workspace is deleted first.
"""

import glob
import os
import shutil
import subprocess
import sys

import vitis

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "sw", "vela_app", "src")
SOURCES = ["main.c", "vela_hw.c", "vela_hw.h", "vela_image.c", "vela_image.h",
           "vela_text.c", "vela_text.h", "golden_ref.h"]
PLATFORM, APP = "vela_platform", "vela_app"
CPU, DOMAIN = "ps7_cortexa9_0", "standalone_ps7_cortexa9_0"


def find_one(pattern):
    found = sorted(glob.glob(pattern, recursive=True))
    if not found:
        raise FileNotFoundError(pattern)
    return found[0]


def main(xsa, workspace, out_dir):
    xsa, workspace, out_dir = (os.path.abspath(p) for p in (xsa, workspace, out_dir))
    shutil.rmtree(workspace, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)

    client = vitis.create_client()
    client.set_workspace(path=workspace)

    # 1. Platform: standalone + FatFs with long file names, FSBL included
    platform = client.create_platform_component(name=PLATFORM, hw_design=xsa, os="standalone",
                                                cpu=CPU, domain_name=DOMAIN)
    domain = platform.get_domain(DOMAIN)
    domain.set_lib(lib_name="xilffs")
    domain.set_config(option="lib", param="XILFFS_use_lfn", value="1", lib_name="xilffs")
    platform.build()
    xpfm = find_one(os.path.join(workspace, PLATFORM, "export", PLATFORM, "*.xpfm"))
    print(f"SWCHECK platform {xpfm}")

    # 2. Application
    app = client.create_app_component(name=APP, platform=xpfm, domain=DOMAIN,
                                      template="empty_application")
    app.import_files(from_loc=SRC, files=SOURCES, dest_dir_in_cmp="src")
    app.append_app_config(key="USER_LINK_LIBRARIES", values=["m"])
    app.build()
    elf = find_one(os.path.join(workspace, APP, "build", f"{APP}.elf"))
    shutil.copy(elf, os.path.join(out_dir, f"{APP}.elf"))
    print(f"SWCHECK elf {elf}")

    # 3. BOOT.BIN for an SD card boot
    fsbl = find_one(os.path.join(workspace, PLATFORM, "**", "fsbl.elf"))
    bit = os.path.join(os.path.dirname(xsa), "vela.bit")
    bif = os.path.join(out_dir, "boot.bif")
    with open(bif, "w") as f:
        f.write("the_ROM_image:\n{\n"
                f"  [bootloader] {fsbl}\n"
                f"  {bit}\n"
                f"  {os.path.join(out_dir, APP + '.elf')}\n"
                "}\n")
    subprocess.run(["bootgen", "-image", bif, "-arch", "zynq", "-o",
                    os.path.join(out_dir, "BOOT.BIN"), "-w", "on"], check=True)
    print(f"SWCHECK fsbl {fsbl}")
    print(f"SWCHECK boot {os.path.join(out_dir, 'BOOT.BIN')}")
    vitis.dispose()


if __name__ == "__main__":
    main(*sys.argv[1:4])
