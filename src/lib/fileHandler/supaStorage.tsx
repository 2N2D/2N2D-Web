"use server"

import {createClient} from '@supabase/supabase-js'

const supaClient = createClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.SUPABASE_SERVICE_KEY!
)

export async function UploadFile(file: File, bucket: string, uid: string, sessionId: number | string) {
    await deleteDirectory(bucket, `${uid}/${sessionId}`);
    const path = `${uid}/${sessionId}/${file.name}`
    console.log(path);
    const fileBuffer = Buffer.from(await file.arrayBuffer());

    const {error} = await supaClient.storage.from(bucket).upload(path, fileBuffer, {
        contentType: file.type,
        upsert: true
    });
    if (error) throw error;

    return path;
}

export async function deleteFile(bucketName: string, filePath: string) {
    try {
        const paths = [filePath];
        await supaClient.storage.from(bucketName).remove(paths);
    } catch (error) {
        console.error('An error occurred:', error);
    }
}

export async function downloadFile(bucketName: string, filePath: string) {
    try {
        console.log(filePath)
        const {data, error} = await supaClient.storage.from(bucketName).download(filePath);
        if (error) {
            console.error('Error downloading file:', error);
            return "202";
        }
        console.log("Download successful")
        return data;
    } catch (error) {
    }
}


async function deleteDirectory(bucketName: string, directoryPath: string) {
    try {
        const {data: objects, error: listError} = await supaClient.storage
            .from(bucketName)
            .list(directoryPath);

        if (listError) {
            console.error('Error listing objects:', listError);
            return;
        }

        if (!objects || objects.length === 0) {
            console.log('Directory is empty or does not exist.');
            return;
        }

        const pathsToDelete = [];
        const subdirectories = [];

        for (const object of objects) {
            if (object.metadata.type === 'file')
                pathsToDelete.push(`${directoryPath}/${object.name}`);
            else if (object.metadata.type === 'directory')
                subdirectories.push(`${directoryPath}/${object.name}`);
            else
                pathsToDelete.push(`${directoryPath}/${object.name}`);

        }

        if (pathsToDelete.length > 0) {
            const {data: deleteData, error: deleteError} = await supaClient.storage
                .from(bucketName)
                .remove(pathsToDelete);

            if (deleteError) {
                console.error('Error deleting objects:', deleteError);
                return;
            }
        }

        for (const subdirectory of subdirectories)
            await deleteDirectory(bucketName, subdirectory);

        console.log('Directory deleted successfully.');
    } catch (error) {
        console.error('An error occurred:', error);
    }
}

