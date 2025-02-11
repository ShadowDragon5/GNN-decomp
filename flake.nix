{
  description = "GNN decomposition packaged using poetry2nix";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    poetry2nix = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.flake-utils.follows = "flake-utils";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      poetry2nix,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        # see https://github.com/nix-community/poetry2nix/tree/master#api for more functions and examples.
        myapp =
          { poetry2nix, lib }:
          poetry2nix.mkPoetryApplication {
            projectDir = self;
            groups = [
              "main"
              "local"
            ];
            overrides = poetry2nix.overrides.withDefaults (
              final: super:
              lib.mapAttrs
                (
                  attr: systems:
                  super.${attr}.overridePythonAttrs (old: {
                    nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ map (a: final.${a}) systems;
                  })
                )
                {
                  # https://github.com/nix-community/poetry2nix/blob/master/docs/edgecases.md#modulenotfounderror-no-module-named-packagename
                  pyg-lib = [
                    "setuptools"
                    "torch"
                  ];
                  torch-sparse = [
                    "setuptools"
                    "torch"
                  ];
                }
            );
          };
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
          overlays = [
            poetry2nix.overlays.default
            (final: _: {
              myapp = final.callPackage myapp { };
            })
          ];
        };
        # libcusparse_lt = pkgs.buildFHSEnv {
        libcusparse_lt = pkgs.stdenv.mkDerivation {
          name = "libcusparse_lt";
          version = "0.7.0.0";
          src = pkgs.fetchzip {
            url = "https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-x86_64/libcusparse_lt-linux-x86_64-0.7.0.0-archive.tar.xz";
            hash = "sha256-4laJj38DmA/BAZj5nYUVd1PXnzRqLknDlJdjNZvq6mU=";
          };
          installPhase = "mkdir -p $out/lib; cp -r lib/* $out/lib";
        };
      in
      {
        packages.default = pkgs.myapp;
        devShells = {
          # Shell for app dependencies.
          #
          #     nix develop
          #
          # Use this shell for developing your app.
          default = pkgs.mkShell {
            inputsFrom = [ pkgs.myapp ];
            buildInputs = [
              libcusparse_lt
            ];
            shellHook = ''
              export LD_LIBRARY_PATH=${libcusparse_lt}/lib:$LD_LIBRARY_PATH
            '';
          };

          # Shell for poetry.
          #
          #     nix develop .#poetry
          #
          # Use this shell for changes to pyproject.toml and poetry.lock.
          poetry = pkgs.mkShell {
            packages = [ pkgs.poetry ];
          };
        };
        legacyPackages = pkgs;
      }
    );
}
