import {User, Session} from "@/lib/sessionHandling/sessionManager"

export let currentUser: User | null = null;
export let currentSession: Session | null = null;

export function setCurrentUser(user: User) {
    currentUser = user;
}

export function setCurrentSession(session: Session) {
    currentSession = session;
}

