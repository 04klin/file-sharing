# file-sharing
decentralized p2p file sharing/storage

Starting template from [Poppyseed Dev](https://merkle-tree-app.vercel.app/merkle-tree/starter-code.html)

## Running the app locally

1. Build the entire workspace
```cargo build --release```

2. Run the server
```cargo run --manifest-path server/Cargo.toml```

3. Run the client setup script
```cargo run --bin setup --manifest-path client/Cargo.toml```

4. Run the main client
```cargo run --bin client --manifest-path client/Cargo.toml http://localhost:8000```