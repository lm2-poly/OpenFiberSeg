import os
from git import Repo

targetPath="./TomographicData"

if not os.path.isdir(targetPath):
    os.mkdir(targetPath)
else:
    raise IOError("Target path already exists at: {}".format(targetPath))

Repo.clone_from("https://github.com/lm2-poly/OpenFiberSeg_TomographicData.git", targetPath)

print("Data pulled from remote repository")
