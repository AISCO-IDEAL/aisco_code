# This is your new script, e.g., main_analysis.py

# Import the *functions* from your other file
import compare

# --- Define your models ---
# Make sure these files are in the same folder
STL_MODEL_PATH = "Jass Box - Lid.stl"
PLY_MODEL_PATH = "Jass Box - Lid.ply"

print("--- Starting Main Analysis Script ---")

# --- 1. Run the Overlap Comparison ---
print("Running overlap test...")
compare.show_overlap(STL_MODEL_PATH, PLY_MODEL_PATH)

# --- 2. Run the Heatmap Comparison ---
print("\nRunning heatmap test...")
# This will color the STL model based on its distance to the PLY model
compare.show_heatmap_comparison(source_file=STL_MODEL_PATH, target_file=PLY_MODEL_PATH)

print("\n--- Analysis Complete ---")
