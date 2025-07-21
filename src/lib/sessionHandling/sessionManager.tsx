"use server"

import {session} from "@/db/schemas/session"
import {user} from "@/db/schemas/user"
import {getSessionTokenHash, getCurrentUser} from "@/lib/auth/authentication";
import {db} from "@/db/db";
import {eq} from "drizzle-orm";
import {deleteFile} from "@/lib/fileHandler/supaStorage";

export type Session = typeof session.$inferSelect;
export type User = typeof user.$inferSelect;

export async function createUser(mail: string, displayName: string): Promise<User> {
    let newUser = await db.insert(user).values({
        uid: await getCurrentUser(),
        email: mail,
        displayName: displayName,
        sessions: []
    }).returning()

    return newUser[0];
}

export async function deleteUser(id: number) {
    await db.delete(user).where(eq(user.id, id));
}

export async function getUser(): Promise<User | string> {
    const userHash = await getCurrentUser();
    if (userHash == "0")
        return "Not logged"
    let User = await db.select().from(user).where(eq(user.uid, userHash));
    return User[0];
}

export async function getSpecificUser(uidHash: string): Promise<User | null> {
    let User = await db.select().from(user).where(eq(user.uid, uidHash));
    if (!User[0] || User[0] == null) return null;
    return User[0];
}

export async function updateUser(newUser: User) {
    await db.update(user).set(newUser).where(eq(user.id, newUser.id));
}

export async function createSession(User: User): Promise<Session> {
    let newSession = await db.insert(session).values({
        userId: await getCurrentUser(),
        tokenHash: await getSessionTokenHash(),
        onnxName: "",
        onnxUrl: "",
        csvName: "",
        csvUrl: "",
        optimizedFileUrl: "",
        visResult: {},
        csvResult: {},
        optResult: {},
        chat: {},
        name: "New Session"
    }).returning()

    return newSession[0];
}

export async function getSession(id: number): Promise<Session> {
    let sessions = await db.select().from(session).where(eq(session.id, id));
    return sessions[0];
}

export async function updateSession(newSession: Session) {
    await db.update(session).set(newSession).where(eq(session.id, newSession.id));
}

export async function deleteSession(id: number) {
    const ses = await getSession(id);
    await db.delete(session).where(eq(session.id, id));
    if (ses.csvUrl && ses.csvUrl.length > 0)
        await deleteFile("csv", ses.csvUrl);

    if (ses.onnxUrl && ses.onnxUrl.length > 0)
        await deleteFile("onnx", ses.onnxUrl);

    if (ses.optimizedFileUrl && ses.optimizedFileUrl.length > 0)
        await deleteFile("rezult", ses.optimizedFileUrl);
}