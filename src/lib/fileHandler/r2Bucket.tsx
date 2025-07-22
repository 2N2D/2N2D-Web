"use server"

import {
  S3Client,
  PutObjectCommand,
  GetObjectCommand,
  DeleteObjectCommand,
  ListBucketsCommand,
  CreateBucketCommand
} from '@aws-sdk/client-s3';

import config from '../../../env.config';
import { Readable } from 'stream';

const client = new S3Client({
  region: 'auto',
  endpoint: config.R2_BUCKET_ENDPOINT,
  credentials: {
    accessKeyId: config.R2_ACCESS_KEY_ID,
    secretAccessKey: config.R2_SECRET_ACCESS_KEY
  },
});

export async function uploadFile(
  file: File,
  path: string,
  bucketName = "2n2d"
) {
  
  const bucktes = await client.send(new ListBucketsCommand())
  if(bucktes.Buckets?.length === 0) 
    await client.send(new CreateBucketCommand({ Bucket: bucketName }));
  const finalPath = `${path}/${file.name}`;
  
  const command = new PutObjectCommand({
    Bucket: bucketName,
    Key: finalPath,
    Body: Buffer.from(await file.arrayBuffer()),
    ContentType: file.type
  });
  
  try {
    await client.send(command);
    console.log(`File uploaded successfully to ${finalPath}`);
    return finalPath;
  } catch (error) {
    if (error) console.log(error);
    throw error;
  }
}

export async function deleteFile(path: string, bucketName = '2n2d') {
  const command = new DeleteObjectCommand({
    Bucket: bucketName,
    Key: path
  });

  try {
    await client.send(command);
  } catch (error) {
    if (error) console.log(error);
  }
}

export async function getFile(path: string, bucketName = '2n2d') {
  const command = new GetObjectCommand({
    Bucket: bucketName,
    Key: path
  });

  try {
    const response = await client.send(command);
    if (!response.Body) return null;
    const chunks: Uint8Array[] = [];
    for await (const chunk of response.Body as Readable) {
      chunks.push(typeof chunk === 'string' ? Buffer.from(chunk) : chunk);
    }
    const buffer = Buffer.concat(chunks);
    return new Blob([buffer]);
  } catch (error) {
    if (error) console.log(error);
    throw error; 
  }
}
