ham_tasks = {
    "Task1": [4,5], # ["nv", "mel"]
    "Task2": [1, 3], # ["bcc", "df"]
    "Task3": [2, 0, 6] # ["bkl", "vasc", "akiec"]
}

d7p_tasks = {
    "Task1": [3,2,0], # ["nv", "mel", "bcc"]
    "Task2": [1,4,5], # ["df", "bkl", "vasc"]
}

dmf_tasks = {
    "Task1": [1,2], # ["nv", "bcc"]
    "Task2": [0, 5], # ["mel", "df"]
    "Task3": [4, 6, 3] # ["bkl", "vasc", "akiec"]
}


IN_DIM_PANDERM, OUT_DIM_PANDERM = 1024, 2048
IN_DIM_DERM, OUT_DIM_DERM = 6144, 10000
IN_DIM_CLIP, OUT_DIM_CLIP = 768, 1024