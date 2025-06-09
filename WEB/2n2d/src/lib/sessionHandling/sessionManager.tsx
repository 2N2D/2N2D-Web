"use server"

import {session} from "@/db/schemas/session"
import {user} from "@/db/schemas/user"
import {getSessionTokenHash, getCurrentUserHash} from "@/lib/auth/authentication";
import {db} from "@/db/db";
import {eq} from "drizzle-orm";

export type Session = typeof session.$inferSelect;
export type User = typeof user.$inferSelect;

export async function createUser(mail: string, displayName: string): Promise<User> {
    let newUser = await db.insert(user).values({
        uid: await getCurrentUserHash(),
        email: mail,
        displayName: displayName,
        sessions: []
    }).returning()

    return newUser[0];
}

export async function deleteUser(id: number) {
    await db.delete(user).where(eq(user.id, id));
}

export async function getUser(): Promise<User> {
    let User = await db.select().from(user).where(eq(user.uid, await getCurrentUserHash()));
    console.log(await getCurrentUserHash());
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
        userId: await getCurrentUserHash(),
        tokenHash: await getSessionTokenHash(),
        onnxName: "",
        onnxUrl: "",
        csvName: "",
        csvUrl: "",
        optimizedFileUrl: "",
        visResult: {},
        csvResult: {},
        optResult: {},
        chat: {}
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
    await db.delete(session).where(eq(session.id, id));
}