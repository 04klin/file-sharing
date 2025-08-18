use actix_web::{web, App, HttpServer, Responder, post, get, HttpResponse};
use serde::{Serialize, Deserialize};
use std::sync::{Mutex, Arc};
use std::collections::HashMap;
use merkle_tree::{hash, generate_proof, validate_proof, HashValue, SiblingNode, MerkleProof};

use aws_sdk_s3::config::Region;
use aws_config::SdkConfig;
use aws_sdk_s3::primitives::ByteStream;
use aws_sdk_s3::Client as S3Client;
use sqlx::{Pool, Postgres, FromRow};

pub struct AppState {
  pub db_pool: Pool<Postgres>,
  pub minio_client: S3Client, // Renamed for clarity
  pub minio_bucket_name: String,
}

// A struct to represent a row in your new 'files' table
#[derive(Serialize, FromRow)]
pub struct FileRecord {
  pub id: i32,
  pub filename: String,
  pub file_hash: String,
}

#[post("/upload")]
async fn upload(file: web::Json<HashMap<String, String>>, state: web::Data<AppState>) -> impl Responder {
  // We get a single file, but your logic supports many. Let's assume one for simplicity.
  if let Some((filename, content)) = file.into_inner().into_iter().next() {
    let file_hash = hash(&content);
    let bucket_name = &state.minio_bucket_name;

    // 1. Upload the file content to MinIO
    let body = ByteStream::from(content.into_bytes());
    match state.minio_client
      .put_object()
      .bucket(bucket_name)
      .key(&filename) // Use the filename as the object key in MinIO
      .body(body)
      .send()
      .await
      {
        Ok(_) => {
          // 2. If upload is successful, save metadata to PostgreSQL
          let query_result = sqlx::query!(
            "INSERT INTO files (filename, file_hash) VALUES ($1, $2) ON CONFLICT (filename) DO UPDATE SET file_hash = $2",
            filename,
            file_hash.to_string()
          )
          .execute(&state.db_pool)
          .await;

          match query_result {
            Ok(_) => HttpResponse::Ok().json(format!("File '{}' uploaded successfully.", filename)),
            Err(e) => {
              eprintln!("Failed to insert file metadata into DB: {}", e);
              HttpResponse::InternalServerError().body("Failed to save file metadata.")
            }
          }
        },
        Err(e) => {
          eprintln!("Failed to upload to MinIO: {}", e);
          HttpResponse::InternalServerError().body("Could not upload file.")
        }
      }
  } else {
    HttpResponse::BadRequest().body("No file data provided.")
  }
}

#[get("/download/{filename}")]
async fn download(path: web::Path<String>, state: web::Data<AppState>) -> impl Responder {
  let filename = path.into_inner();
  
  // 1. Check if the file metadata exists in the database
  let query_result = sqlx::query_as::<_, FileRecord>("SELECT id, filename, file_hash FROM files WHERE filename = $1")
    .bind(&filename)
    .fetch_optional(&state.db_pool)
    .await;

  match query_result {
    Ok(Some(_file_record)) => {
      // 2. If it exists, fetch the object from MinIO
      match state.minio_client
        .get_object()
        .bucket(&state.minio_bucket_name)
        .key(&filename)
        .send()
        .await
      {
        Ok(output) => {
          // 3. Collect the stream into a byte buffer
          let body_bytes = match output.body.collect().await {
            Ok(agg) => agg.into_bytes(),
            Err(_) => return HttpResponse::InternalServerError().body("Failed to read file stream from storage."),
          };

          // 4. Send the collected bytes in the response body
          HttpResponse::Ok()
            .content_type("application/octet-stream")
            .body(body_bytes)
        },
        Err(_) => HttpResponse::InternalServerError().body("Could not retrieve file from storage."),
      }
    },
    Ok(None) => HttpResponse::NotFound().finish(),
    Err(_) => HttpResponse::InternalServerError().body("Error querying database."),
  }
}

#[derive(Serialize)]
struct ProofResponse {
  root: HashValue,
  proof: MerkleProof,
}

#[get("/proof/{filename}")]
async fn proof(path: web::Path<String>, state: web::Data<AppState>) -> impl Responder {
  let filename = path.into_inner();

  // 1. Fetch all file hashes from the database in a consistent order
  let hashes_result = sqlx::query_scalar::<_, String>(
    "SELECT file_hash FROM files ORDER BY filename ASC"
  )
  .fetch_all(&state.db_pool)
  .await;

  let all_hashes = match hashes_result {
    Ok(hashes) => {
      if hashes.is_empty() {
        return HttpResponse::NotFound().body("No files exist to generate a proof.");
      }
      hashes
    },
    Err(_) => return HttpResponse::InternalServerError().body("Could not query file hashes."),
  };
  
  // 2. We also need to get the specific file's info to find its index
  let file_record_result = sqlx::query_as::<_, FileRecord>(
    "SELECT id, filename, file_hash FROM files WHERE filename = $1"
  )
  .bind(&filename)
  .fetch_optional(&state.db_pool)
  .await;

  if let Ok(Some(file_record)) = file_record_result {
    // 3. Find the index of our file's hash in the sorted list
    let index = all_hashes.iter().position(|h| h == &file_record.file_hash);

    if let Some(idx) = index {
      // 4. Join the hashes into a single string for the merkle_tree library
      let concatenated_hashes = all_hashes.join(" ");

      // 5. Generate the proof and root on-demand
      let (generated_root, proof) = generate_proof(&concatenated_hashes, idx);
      
      let response = ProofResponse {
        root: generated_root,
        proof: proof,
      };

      HttpResponse::Ok().json(response)
    } else {
      // This should theoretically not happen if the data is consistent
      HttpResponse::InternalServerError().body("Hash not found in the sorted list.")
    }

  } else {
    HttpResponse::NotFound().body("File not found.")
  }
}

#[get("/hello")]
async fn hello() -> impl Responder {
  HttpResponse::Ok().body("Hello, World!")
}

pub fn create_app_state(pool: Pool<Postgres>, client: aws_sdk_s3::Client, bucket_name: &str) -> web::Data<AppState> {
  web::Data::new(AppState {
    db_pool: pool,
    minio_client: client,
    minio_bucket_name: bucket_name.to_string(),
  })
}

pub fn configure_services(cfg: &mut web::ServiceConfig) {
  cfg.service(upload);
  cfg.service(download);
  cfg.service(proof);
  cfg.service(hello);
}