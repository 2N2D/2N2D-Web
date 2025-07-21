import {
  S3Client,
  PutObjectCommand,
  GetObjectCommand,
  DeleteObjectCommand
} from '@aws-sdk/client-s3';

import config from '../../../env.config';

const client = new S3Client({
  region: 'auto',
  endpoint: config.R2_BUCKET_ENDPOINT,
  credentials: {
    accessKeyId: config.R2_ACCESS_KEY_ID,
    secretAccessKey: config.R2_SECRET_ACCESS_KEY
  }
});

export async function uploadFile(
  file: File,
  path: string,
  bucketName = '2n2d'
) {
  const finalPath = `${path}/${file.name}`;

  const command = new PutObjectCommand({
    Bucket: bucketName,
    Key: finalPath,
    Body: file,
    ContentType: file.type
  });

  try {
    await client.send(command);
  } catch (error) {
    if (error) console.log(error);
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
    return response.Body;
  } catch (error) {
    if (error) console.log(error);
    throw error; 
  }
}
