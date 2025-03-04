{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
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
        pkgs = import nixpkgs { inherit system; };
        venvDir = ".venv";
      in
      {
        # `nix develop`
        devShell = pkgs.mkShell {
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
            export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib.outPath}/lib:${pkgs.pythonManylinuxPackages.manylinux2014Package}/lib:$LD_LIBRARY_PATH";
            if ! [ -d "${venvDir}" ]; then
              ${pkgs.python3.interpreter} -m venv "${venvDir}"
              source "${venvDir}/bin/activate"
              pip install -r requirements.txt
            fi
            source "${venvDir}/bin/activate"
          '';
        };
      }
    );
}
