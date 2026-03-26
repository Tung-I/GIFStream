import os
import subprocess
from argparse import ArgumentParser
import sys
import shutil
import glob
from pre_colmap import COLMAPDatabase
import numpy as np

def posetow2c_matrcs(poses):
    tmp = inversestep4(inversestep3(inversestep2(inversestep1(poses))))
    N = tmp.shape[0]
    ret = []
    for i in range(N):
        ret.append(tmp[i])
    return ret

def inversestep4(c2w_mats):
    return np.linalg.inv(c2w_mats)
def inversestep3(newposes):
    tmp = newposes.transpose([2, 0, 1]) # 20, 3, 4 
    N, _, __ = tmp.shape
    zeros = np.zeros((N, 1, 4))
    zeros[:, 0, 3] = 1
    c2w_mats = np.concatenate([tmp, zeros], axis=1)
    return c2w_mats

def inversestep2(newposes):
    return newposes[:,0:4, :]
def inversestep1(newposes):
    poses = np.concatenate([newposes[:, 1:2, :], newposes[:, 0:1, :], -newposes[:, 2:3, :],  newposes[:, 3:4, :],  newposes[:, 4:5, :]], axis=1)
    return poses

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def convertdynerftocolmapdb(path, offset=0):
    originnumpy = os.path.join(path, "poses_bounds.npy")
    mp4_path = os.path.join(path,"mp4")
    projectfolder = os.path.join(path, "colmap_" + str(offset))
    #sparsefolder = os.path.join(projectfolder, "sparse/0")
    manualfolder = os.path.join(projectfolder, "manual")

    # if not os.path.exists(sparsefolder):
    #     os.makedirs(sparsefolder)
    if not os.path.exists(manualfolder):
        os.makedirs(manualfolder)

    savetxt = os.path.join(manualfolder, "images.txt")
    savecamera = os.path.join(manualfolder, "cameras.txt")
    savepoints = os.path.join(manualfolder, "points3D.txt")
    imagetxtlist = []
    cameratxtlist = []
    if os.path.exists(os.path.join(projectfolder, "input.db")):
        os.remove(os.path.join(projectfolder, "input.db"))

    db = COLMAPDatabase.connect(os.path.join(projectfolder, "input.db"))

    db.create_tables()


    with open(originnumpy, 'rb') as numpy_file:
        poses_bounds = np.load(numpy_file)
        poses = poses_bounds[:, :15].reshape(-1, 3, 5)

        llffposes = poses.copy().transpose(1,2,0)
        w2c_matriclist = posetow2c_matrcs(llffposes)
        assert (type(w2c_matriclist) == list)


        for i in range(len(poses)):
            cameraname = f"cam{i:02d}" #"cam" + str(i).zfill(2)
            m = w2c_matriclist[i]
            colmapR = m[:3, :3]
            T = m[:3, 3]
            
            H, W, focal = poses[i, :, -1]
            
            colmapQ = rotmat2qvec(colmapR)
            # colmapRcheck = qvec2rotmat(colmapQ)

            imageid = str(i+1)
            cameraid = imageid
            pngname = cameraname + ".png"
            
            line =  imageid + " "

            for j in range(4):
                line += str(colmapQ[j]) + " "
            for j in range(3):
                line += str(T[j]) + " "
            line = line  + cameraid + " " + pngname + "\n"
            empltyline = "\n"
            imagetxtlist.append(line)
            imagetxtlist.append(empltyline)

            focolength = focal
            model, width, height, params = i, W, H, np.array((focolength,  focolength, W//2, H//2,))

            camera_id = db.add_camera(1, width, height, params)
            cameraline = str(i+1) + " " + "PINHOLE " + str(width) +  " " + str(height) + " " + str(focolength) + " " + str(focolength) + " " + str(W//2) + " " + str(H//2) + "\n"
            cameratxtlist.append(cameraline)
            
            image_id = db.add_image(pngname, camera_id,  prior_q=np.array((colmapQ[0], colmapQ[1], colmapQ[2], colmapQ[3])), prior_t=np.array((T[0], T[1], T[2])), image_id=i+1)
            db.commit()
        db.close()


    with open(savetxt, "w") as f:
        for line in imagetxtlist :
            f.write(line)
    with open(savecamera, "w") as f:
        for line in cameratxtlist :
            f.write(line)
    with open(savepoints, "w") as f:
        pass 

def getcolmapsinglen3d(folder, offset):
    
    folder = os.path.join(folder, "colmap_" + str(offset))
    assert os.path.exists(folder)

    dbfile = os.path.join(folder, "input.db")
    inputimagefolder = os.path.join(folder, "input")
    distortedmodel = os.path.join(folder, "distorted/sparse")
    step2model = os.path.join(folder, "tmp")
    if not os.path.exists(step2model):
        os.makedirs(step2model)

    manualinputfolder = os.path.join(folder, "manual")
    if not os.path.exists(distortedmodel):
        os.makedirs(distortedmodel)

    featureextract = "colmap feature_extractor --database_path " + dbfile+ " --image_path " + inputimagefolder + " --SiftExtraction.max_image_size 4096 --SiftExtraction.max_num_features 106384 --SiftExtraction.estimate_affine_shape 1 --SiftExtraction.domain_size_pooling 1 --ImageReader.camera_model PINHOLE"

    exit_code = os.system(featureextract)
    if exit_code != 0:
        exit(exit_code)

    featurematcher = "colmap exhaustive_matcher --database_path " + dbfile
    exit_code = os.system(featurematcher)
    if exit_code != 0:
        exit(exit_code)

   # threshold is from   https://github.com/google-research/multinerf/blob/5b4d4f64608ec8077222c52fdf814d40acc10bc1/scripts/local_colmap_and_resize.sh#L62
    triandmap = "colmap point_triangulator --database_path "+   dbfile  + " --image_path "+ inputimagefolder + " --output_path " + distortedmodel \
    + " --input_path " + manualinputfolder + " --Mapper.ba_global_function_tolerance=0.000001"
   
    exit_code = os.system(triandmap)
    if exit_code != 0:
       exit(exit_code)
    print(triandmap)


    img_undist_cmd = "colmap" + " image_undistorter --image_path " + inputimagefolder + " --input_path " + distortedmodel  + " --output_path " + folder  \
    + " --output_type COLMAP" 
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        exit(exit_code)
    print(img_undist_cmd)

    removeinput = "rm -r " + inputimagefolder
    exit_code = os.system(removeinput)
    if exit_code != 0:
        exit(exit_code)

    files = os.listdir(folder + "/sparse")
    os.makedirs(folder + "/sparse/0", exist_ok=True)
    for file in files:
        if file == '0':
            continue
        source_file = os.path.join(folder, "sparse", file)
        destination_file = os.path.join(folder, "sparse", "0", file)
        shutil.move(source_file, destination_file)

if __name__ == "__main__" :
    parser = ArgumentParser(description="dataset information")
    parser.add_argument("--root_dir", type=str, default = None)
    parser.add_argument("--extract_frames", action='store_true')
    parser.add_argument("--frame_rate", type=int, default = 30)
    parser.add_argument("--startframe", default=0, type=int)
    parser.add_argument("--endframe", default=300, type=int)
    parser.add_argument("--GOP", default=60, type=int)
    args = parser.parse_args(sys.argv[1:])

    for folder_name in os.listdir(args.root_dir):
        source_path = os.path.join(args.root_dir, folder_name)
        output_path = os.path.join(args.root_dir, folder_name, "png")
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        
        # extract frams from videos
        if os.path.isdir(source_path) and args.extract_frames:
            mp4_files = sorted([name for name in os.listdir(source_path) if name.endswith(".mp4")])
            for idx,file_name in enumerate(mp4_files):
                video_path = os.path.join(source_path, file_name)
                output_folder = os.path.join(output_path, f"cam{idx:02d}")
                if not os.path.exists(output_path):
                        os.mkdir(output_path)
                
                if not os.path.exists(output_folder):
                    os.mkdir(output_folder)
                
                cmd = f"ffmpeg -i {video_path} -vf fps={args.frame_rate} {output_folder}/%05d.png"
                subprocess.call(cmd, shell=True)
        
        # conduct colmap for GOPs
        camera_names = [name for name in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, name))]
        camera_names = sorted(camera_names)
        frame_list = [args.startframe + x * args.GOP for x in range((args.endframe-args.startframe + args.GOP-1)//args.GOP)]
        for frame in frame_list:
            colmap_path = os.path.join(args.root_dir, folder_name, f"colmap_{frame}")
            first_frame_path = os.path.join(args.root_dir, folder_name, f"colmap_{frame}", "input")
            if not os.path.exists(colmap_path):
                os.mkdir(colmap_path)
            if not os.path.exists(first_frame_path):
                os.mkdir(first_frame_path)
            for ind,cam in enumerate(camera_names):
                image_path = os.path.join(output_path, cam, f"{(frame+1):05d}.png")
                save_path = os.path.join(first_frame_path, f"cam{ind:02d}.png")
                shutil.copy(image_path, save_path)

            convertdynerftocolmapdb(os.path.join(args.root_dir, folder_name),frame)
            getcolmapsinglen3d(os.path.join(args.root_dir, folder_name),frame)
