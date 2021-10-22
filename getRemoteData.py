import os
from git import Repo

targetPath="./test"#"./TomographicData"

if not os.path.isdir(targetPath):
    os.mkdir(targetPath)
else:
    raise IOError("Target path already exists at: {}".format(targetPath))

Repo.clone_from("https://github.com/FacundoLM2/RPi_timelapse", targetPath)

print("Data pulled from remote repository")