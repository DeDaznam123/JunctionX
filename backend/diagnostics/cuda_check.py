import os, ctypes, json, sys

def find_dll(name: str):
    hits = []
    for p in os.environ.get("PATH", "").split(os.pathsep):
        if not p or not os.path.isdir(p):
            continue
        try:
            for f in os.listdir(p):
                if f.lower() == name.lower():
                    hits.append(os.path.join(p, f))
        except Exception:
            pass
    return hits

def main():
    info = {}
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["torch_cuda_version"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        info["device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if torch.cuda.is_available():
            info["device_0_name"] = torch.cuda.get_device_name(0)
    except Exception as e:
        info["torch_error"] = str(e)

    dll = "cudnn_ops64_9.dll"
    dll_paths = find_dll(dll)
    info["cudnn_ops64_9_locations"] = dll_paths
    if dll_paths:
        try:
            ctypes.CDLL(dll)
            info["cudnn_load"] = "ok"
        except Exception as e:
            info["cudnn_load"] = f"failed: {e}"
    else:
        info["cudnn_load"] = "not found"

    print(json.dumps(info, indent=2))

if __name__ == "__main__":
    main()
