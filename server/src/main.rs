use std::env;
use actix_web::{App, HttpServer};
use server::{create_app_state, configure_services};
use sqlx::{postgres::PgPoolOptions, Pool, Postgres, migrate};

use aws_config::SdkConfig;
use aws_sdk_s3::{Client, config::Region};

#[actix_web::main]
async fn main() -> std::io::Result<()> {

  // 1. Load environment vars
  dotenvy::dotenv().ok();

  // 2. Setup PostgreSQL connection pool
  let database_url: String = env::var("DATABASE_URL")
    .expect("DATABASE_URL must be set in .env file");
  
  let pool: Pool<Postgres> = PgPoolOptions::new()
    .max_connections(5)
    .connect(&database_url)
    .await
    .expect("Failed to create PostgreSQL connection pool");

  sqlx::migrate!("./db")
    .run(&pool)
    .await
    .expect("Failed to run migrations!");

  // 3. Setup MinIO client
  let sdk_config: SdkConfig = aws_config::from_env()
    .region(Region::new("us-east-1"))
    .endpoint_url("http://minio:9000") // This is the crucial line for MinIO
    .load()
    .await;

  let s3_client = Client::new(&sdk_config);

  // 4. Create the MinIO bucket if it doesn't exist
  let bucket_name = "my-rust-bucket";
  match s3_client.head_bucket().bucket(bucket_name).send().await {
    Ok(_) => println!("Bucket '{}' already exists.", bucket_name),
    Err(_) => {
      println!("Creating bucket '{}'...", bucket_name);
      s3_client
        .create_bucket()
        .bucket(bucket_name)
        .send()
        .await
        .expect("Failed to create MinIO bucket.");
    }
  };

  // 4. Create the Actix-web server
  let state: actix_web::web::Data<server::AppState> = create_app_state(pool, s3_client, bucket_name);

  HttpServer::new(move || {
    App::new()
      // 5. Add the database pool and minio bucket to the application state
      .app_data(state.clone())
      .configure(configure_services)
  })
  .bind("0.0.0.0:8000")?
  .run()
  .await
}