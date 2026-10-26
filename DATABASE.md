# LogInsight Database Overview

LogInsight uses SQLite databases without a database server. Data is split across three types of database files: a live monitor database, an authentication database, and batch run databases.

---

## App Usage and Database Files

* **`backend/monitor.sqlite3`**: Used by the live monitor. Contains live monitoring data, traffic, IP activity, and tracking metadata.
* **`backend/auth.sqlite3`**: Used for logins (user management and active sessions).
* **`backend/runs/<job_id>.sqlite3`**: Used for batch runs of a test set (currently 15 files exist). Each batch run file contains only the `blocks` and `explanations` tables using the same schema as the live monitor.

---

## Database Schemas

### 1. `monitor.sqlite3`

#### Table: `blocks`
* **Row count:** 115
* **Indexes:** `blocks_last_seen(last_seen)`
* **Columns:**
  * `block_id` (TEXT, Primary Key): One row per HDFS block.
  * `anomaly_score` (REAL): LSTM score ranging from 0 to 1.
  * `is_anomalous` (INTEGER): Set to `1` if the score is greater than 0.5.
  * `sequence_len` (INTEGER): Length of the sequence.
  * `event_ids` (TEXT): JSON list of template IDs (maximum 50 used).
  * `first_seen` (INTEGER): Unix timestamp in seconds (log timestamps treated as UTC).
  * `last_seen` (INTEGER): Unix timestamp in seconds (log timestamps treated as UTC).

#### Table: `explanations`
* **Row count:** 2
* **Columns:**
  * `block_id` (TEXT, Primary Key)
  * `seq_len` (INTEGER)
  * `body` (TEXT): Gemini JSON answer containing summary, likely_cause, confidence, incomplete_sequence, and suggested_checks. Acts as a cache so a block is explained once per sequence length.

#### Table: `traffic`
* **Row count:** 307
* **Indexes:** `traffic_t(t)`
* **Primary Key:** None
* **Columns:**
  * `t` (INTEGER): Unix seconds.
  * `src` (TEXT): IP address.
  * `bytes` (INTEGER): One row per block transfer line.

#### Table: `ip_activity`
* **Row count:** 3099
* **Primary Key:** None
* **Columns:**
  * `t` (INTEGER): Unix seconds.
  * `ip` (TEXT): IP address.
  * `n` (INTEGER): Count of log lines mentioning that IP in that second.

#### Table: `meta`
* **Row count:** 4
* **Columns:**
  * `key` (TEXT, Primary Key): Configuration/status keys (`offset` for byte position in the watched log, `lines` for lines read, `unmatched` for lines matching no template, `file` for path watched).
  * `value` (TEXT): The value associated with the key.

---

### 2. `auth.sqlite3`

#### Table: `users`
* **Columns:**
  * `id` (INTEGER, Primary Key)
  * `email` (TEXT, Unique)
  * `salt` (BLOB)
  * `hash` (BLOB): Salted password hash.
  * `created` (REAL)

#### Table: `sessions`
* **Columns:**
  * `token_hash` (TEXT, Primary Key): Hash of the login token.
  * `user_id` (INTEGER): Refers to `users.id` (no declared foreign key).
  * `expires` (REAL)

---

## How Tables Relate

* **`blocks` to `explanations`**: `blocks.block_id` matches `explanations.block_id`. This is a one-to-one relationship (not enforced by a foreign key constraint).
* **`sessions` to `users`**: `sessions.user_id` references `users.id` (not enforced by a foreign key constraint).
* **Unrelated tables**: `traffic` and `ip_activity` are not linked to blocks.

