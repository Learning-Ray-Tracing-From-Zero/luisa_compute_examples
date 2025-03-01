import subprocess
import argparse


def main():
    parser = argparse.ArgumentParser(description="Run ray tracing with different scenes.")
    parser.add_argument("--backend", type=str, default="cuda", help="Backend to use (e.g., cuda, cpu).")
    parser.add_argument("--samples", type=int, default=1, help="Number of samples per pixel.")
    args = parser.parse_args()

    command_template = "./build/release/bin/ray_tracing_in_one_weekend --backend {backend} --samples {samples} --outfile ./test_{scene} --scene {scene}"
    for scene in range(2, 8):
        command = command_template.format(
            backend=args.backend,
            samples=args.samples,
            scene=scene
        )
        print(f"Running command: {command}")
        
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Scene {scene} completed successfully.")
        else:
            print(f"Scene {scene} failed with error:")
            print(result.stderr)


if __name__ == "__main__":
    main()
