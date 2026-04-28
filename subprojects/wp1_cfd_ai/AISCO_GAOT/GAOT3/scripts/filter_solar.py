import pandas as pd

# Pfade
input_file = "/cluster/work/fuge/AISCO/wp1_cfd_ai/blended_wing/test/case_data.dat"
output_file = "/cluster/work/fuge/AISCO/wp1_cfd_ai/blended_wing/test/case_data_solar.dat"

# Einlesen mit dem flexiblen Trenner r'\s+' (fängt Tabs und Leerzeichen ab)
df = pd.read_csv(
    input_file,
    sep=r"\s+",
    comment="#",
    names=["case_name", "geom_name", "alt_kft", "Re", "M_inf", "alpha_deg", "beta_deg", "CD", "CL", "CMy"],
)

# Filter: Mach <= 0.2 und Alpha zwischen -2 und +5 Grad
# Wir lassen Altitude hier weg, um mehr Geometrie-Varianz zu behalten,
# da Cp normiert ist [cite: 161]
mask = (df["M_inf"] <= 0.1) & (df["alpha_deg"] >= -2.0) & (df["alpha_deg"] <= 5.0)
df_filtered = df[mask]

# Speichern
with open(output_file, "w") as f:
    f.write("# case_name geom_name alt_kft Re M_inf alpha_deg beta_deg CD CL CMy\n")
    df_filtered.to_csv(f, sep="\t", index=False, header=False)

print(f"Fertig! {len(df_filtered)} von {len(df)} Cases behalten.")
