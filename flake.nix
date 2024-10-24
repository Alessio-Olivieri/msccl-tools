{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    poetry2nix.url = "github:nix-community/poetry2nix";
  };

  outputs = { self, nixpkgs, poetry2nix }:
    let 
    system = "x86_64-linux";
    pkgs = nixpkgs.legacyPackages.${system};
    in
   {
    packages.${system}.default = pkgs.mkShell{
      buildinputs=[
        pkgs.python3
        poetry2nix.mkPoetryEnv{
            projectDir = ./.;
            editable = true;   
        }
      ];
    };
};
}
