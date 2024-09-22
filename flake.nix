{
  description = "A Python environment for msccl-tools";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }: {
    devShells.default = nixpkgs.lib.mkShell {
      packages = [
        nixpkgs.python310
        nixpkgs.python310Packages.pip
        nixpkgs.git
      ];
    };
  };
}
