GUI assembly workflow
=====================

This is a short workflow of A) how to use the assembly maker for the GUI, B) How to add assemblies to
    the assembly map, and C) what you can change in the GUI_assembly.py file


A) How to use the assembly maker for the GUI:

    1) First update:

        parameters.yaml

        Here you can change any coordinte, or add more wheels (however many you want, no limit).
            Just be carefull to name them wheel_x , where x is the number of the wheel.

    2) Then call:

        call_assembly_from_params()

    This reads parameters.yaml, generates the assembly, and writes:

        assembly.ply
        assembly.step
        checks.txt

    3) Grab assembly.ply or assembly.step for the generated geometry. Read checks.txt to see the latest
        validity check result.

    Thats it. now you can see the generated assembly in the ply/step file and the validity check in the txt file.
        Note: The assembly doesnt have to be valid for the ply file to get generated. The validity check is
        important later.

B) If you want to add this new assembly to the assembly map so that WP4 (and the rest of the project)
        has access to it, call:

        add_current_gui_assembly_to_map()

    If the assembly is valid, it is added to the map and the function returns the
    new assembly ID.

    If the assembly is not valid, the function returns None and does not add it to
    the map. Check checks.txt to see why it failed.


C) Things you can change in GUI_assembly.py:

    - _EXPORT_* flags choose what gets exported into assembly.ply and assembly.step. (Seat, wheels, outer, etc)
    - _CHECK_* flags enable or disable checks before adding to assembly_map.yaml.
