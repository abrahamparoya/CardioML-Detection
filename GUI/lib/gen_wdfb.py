import numpy as np

def create_blank_dat(filename, num_leads, num_samples):
    # Create a blank dat file with zeros
    data = np.zeros((num_leads, num_samples))
    np.savetxt(filename, data, fmt='%d')

def create_blank_hea(filename, num_leads):
    # Create a blank hea file with appropriate metadata
    with open(filename, 'w') as f:
        f.write("12\n")  # Number of leads
        for i in range(1, num_leads + 1):
            f.write(f"D{i} WAVEFORM SIGNAL 0.0 1000.0 1000 16 + 0 {i} -1\n")
        f.write("#Age: 0  Sex: M  Dx: \n")  # Additional patient information (can be modified as needed)

# Define the parameters
num_leads = 12
num_samples = 1000
dat_filename = "blank_record.dat"
hea_filename = "blank_record.hea"

# Generate blank .dat and .hea files
create_blank_dat(dat_filename, num_leads, num_samples)
create_blank_hea(hea_filename, num_leads)

print(f"Blank .dat file ({dat_filename}) and .hea file ({hea_filename}) have been created.")

