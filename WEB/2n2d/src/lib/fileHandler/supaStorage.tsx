"use server"

import {createClient} from '@supabase/supabase-js'

const supaClient = createClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.SUPABASE_SERVICE_KEY!
)

export async function UploadFile(file: File, bucket: string, uid: string, sessionId: number | string) {
    const path = `${uid}/${sessionId}/${file.name}`
    const fileBuffer = Buffer.from(await file.arrayBuffer());

    const {error} = await supaClient.storage.from(bucket).upload(path, fileBuffer, {
        contentType: file.type,
        upsert: true
    });
    if (error) throw error;

    return path;
}