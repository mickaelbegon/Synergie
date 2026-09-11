## Movella DOT driver and SDK review

Official sources checked:

- Movella software download page
- Movella DOT product page
- Movella DOT 2023.6.0 release article

Key findings:

1. Movella still distributes the Movella DOT PC SDK for Windows and Linux from the official software portal.
2. The public Movella release article for DOT 2023.6.0 states a compatibility break:
   - firmware 3.0.0 and above should be paired with SDK/app 2023.6.0 and above
   - firmware below that line has separate compatibility expectations
3. The official download page still distinguishes the Data Exporter by firmware level:
   - below 2.4.0
   - 2.4.0 and above

Code impact in this repository:

- The previous code hard-coded the Python binding module `movelladot_pc_sdk_py39_64`.
- The project environment now targets Python 3.11.
- A dynamic loader has been added so the application can try:
  - `movelladot_pc_sdk_py311_64`
  - `movelladot_pc_sdk_py310_64`
  - `movelladot_pc_sdk_py39_64`

Recommended operational check on a workstation with the SDK installed:

1. Verify the installed Movella DOT PC SDK version.
2. Verify the installed binding module name under the local `movelladot_pc_sdk` package.
3. Verify the DOT firmware generation in use.
4. Align firmware and exporter choice with the Movella compatibility notes.

Recommended next maintenance step:

- Recreate the project Conda environment and run a real hardware smoke test with at least one DOT over Bluetooth and one USB export path.
