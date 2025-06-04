CREATE TABLE "session" (
	"id" serial PRIMARY KEY NOT NULL,
	"tokenhash" text NOT NULL,
	"userid" text NOT NULL,
	"onnxname" text,
	"onnxurl" text,
	"csvname" text,
	"csvurl" text,
	"optimizedfileurl" text,
	"visresult" json,
	"csvresult" json,
	"optresult" json,
	"chat" json
);
--> statement-breakpoint
CREATE TABLE "user" (
	"id" serial PRIMARY KEY NOT NULL,
	"uid" text NOT NULL,
	"email" text NOT NULL,
	"displayname" text,
	"sessions" text[]
);
