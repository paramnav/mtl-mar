# Set this to your conflicted notebook
conflicted_path = "InfoGain_Ever.ipynb"

with open(conflicted_path, "r") as f:
    lines = f.readlines()

head_lines = []
remote_lines = []

in_head = False
in_remote = False

for line in lines:
    if line.startswith("<<<<<<<"):
        in_head = True
        continue
    elif line.startswith("======="):
        in_head = False
        in_remote = True
        continue
    elif line.startswith(">>>>>>>"):
        in_remote = False
        continue

    if in_head:
        head_lines.append(line)
    elif in_remote:
        remote_lines.append(line)
    else:
        head_lines.append(line)
        remote_lines.append(line)

with open("InfoGain_Ever_HEAD.ipynb", "w") as f:
    f.writelines(head_lines)

with open("InfoGain_Ever_remote.ipynb", "w") as f:
    f.writelines(remote_lines)
    
print("Conflict split complete: InfoGain_Ever_HEAD.ipynb and InfoGain_Ever_remote.ipynb created.")
