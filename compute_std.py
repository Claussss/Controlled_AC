import os
import glob
import math
import torch

def main():
    # Set input and output directories (adjust as needed)
    input_dir = "/home/yurii/Projects/AC/your_pt_files_dir"  # directory containing .pt files
    output_dir = "/home/yurii/Projects/AC/your_std_output_dir"  # directory to save std files
    os.makedirs(output_dir, exist_ok=True)
    
    pt_files = glob.glob(os.path.join(input_dir, "*.pt"))
    print(f"Found {len(pt_files)} .pt files.")

    # Initialize per-dimension accumulators (256 dimensions)
    content_sum = torch.zeros(256, dtype=torch.double)
    content_sumsq = torch.zeros(256, dtype=torch.double)
    content_count = torch.zeros(256, dtype=torch.double)

    prosody_sum = torch.zeros(256, dtype=torch.double)
    prosody_sumsq = torch.zeros(256, dtype=torch.double)
    prosody_count = torch.zeros(256, dtype=torch.double)

    for file in pt_files:
        try:
            data = torch.load(file)
            if not isinstance(data, dict):
                print(f"Skipping {file}: not a dictionary.")
                continue
            content = data.get("content", None)
            prosody = data.get("prosody", None)
            if content is None or prosody is None:
                print(f"Skipping {file}: missing 'content' or 'prosody' key.")
                continue
            # Verify the tensor shape: expecting [256, T]
            if content.shape[0] != 256:
                print(f"Skipping {file}: 'content' tensor does not have 256-dim in axis 0.")
                continue
            T = content.size(1)
            content_sum += content.double().sum(dim=1)
            content_sumsq += (content.double() ** 2).sum(dim=1)
            content_count += T

            if prosody.shape[0] != 256:
                print(f"Skipping {file}: 'prosody' tensor does not have 256-dim in axis 0.")
                continue
            T_p = prosody.size(1)
            prosody_sum += prosody.double().sum(dim=1)
            prosody_sumsq += (prosody.double() ** 2).sum(dim=1)
            prosody_count += T_p
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue

    # Compute per-dimension variance and standard deviation (unbiased)
    std_content = torch.sqrt((content_sumsq - (content_sum ** 2) / content_count) / (content_count - 1))
    std_prosody = torch.sqrt((prosody_sumsq - (prosody_sum ** 2) / prosody_count) / (prosody_count - 1))

    print("Computed std for content per dimension:", std_content)
    print("Computed std for prosody per dimension:", std_prosody)

    torch.save(std_content, os.path.join(output_dir, "std_content.pt"))
    torch.save(std_prosody, os.path.join(output_dir, "std_prosody.pt"))
    print(f"Saved std files to {output_dir}")

if __name__ == "__main__":
    main()
