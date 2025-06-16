{
  nixConfig = {
    extra-substituters = "https://cachix.cachix.org";
    extra-trusted-public-keys = "cachix.cachix.org-1:eWNHQldwUO7G2VkjpnjDbWwy4KQ/HNxht7H4SSoMckM=";
  };

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      nixpkgs,
      flake-utils,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        venvDir = ".venv";
      in
      {
        # CPU-only shell
        # `nix develop`
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            python3
            python3Packages.pip
            python3Packages.virtualenv
            pythonManylinuxPackages.manylinux2014Package
            cmake
            ninja
            imagemagick
          ];

          shellHook = ''
            export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib.outPath}/lib:${pkgs.pythonManylinuxPackages.manylinux2014Package}/lib:$LD_LIBRARY_PATH"
            if ! [ -d "${venvDir}" ]; then
              ${pkgs.python3.interpreter} -m venv "${venvDir}"
              source "${venvDir}/bin/activate"
              pip install -r requirements.txt
            else
              source "${venvDir}/bin/activate"
            fi
          '';
        };

        # CUDA shell
        # `nix develop .#cuda`
        devShells.cuda = pkgs.mkShell {
          buildInputs = with pkgs; [
            python3
            (python3.withPackages (ps: with ps; [ torchWithCuda ]))

            cudatoolkit
            cudaPackages.cudnn
            cudaPackages.cuda_cudart

            # CUDA 12 still wants a GCC ≤ 13 for nvcc
            gcc13
          ];

          shellHook = ''
            export CUDA_PATH=${pkgs.cudatoolkit}

            export CC=${pkgs.gcc13}/bin/gcc
            export CXX=${pkgs.gcc13}/bin/g++
            export PATH=${pkgs.gcc13}/bin:$PATH

            export LD_LIBRARY_PATH="${
              pkgs.lib.makeLibraryPath [
                pkgs.cudatoolkit
                pkgs.cudaPackages.cudnn
              ]
            }:$LD_LIBRARY_PATH"

            export LIBRARY_PATH="${pkgs.lib.makeLibraryPath [ pkgs.cudatoolkit ]}:$LIBRARY_PATH"

            if [ ! -d "${venvDir}" ]; then
              ${pkgs.python3.interpreter} -m venv "${venvDir}"
              source "${venvDir}/bin/activate"
              pip install -r requirements.txt
            else
              source "${venvDir}/bin/activate"
            fi
          '';
        };
      }
    );
}
