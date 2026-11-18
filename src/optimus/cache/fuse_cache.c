#define FUSE_USE_VERSION 31

#include <fuse3/fuse.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <sys/statvfs.h>
#include <sys/time.h>
#include <time.h>
#include <sqlite3.h>
#include <openssl/sha.h>
#include <stddef.h>

#define CHUNK_SIZE 65536

struct cache_options {
    const char *origin;
    const char *cache_dir;
    const char *temp_dir;
    const char *locks_dir;
    const char *db_path;
    const char *dataset_id;
    const char *dataset_version;
    int max_capacity_gb;
    int show_help;
};

static struct cache_options options = {
    .origin = NULL,
    .cache_dir = NULL,
    .temp_dir = NULL,
    .locks_dir = NULL,
    .db_path = NULL,
    .dataset_id = NULL,
    .dataset_version = NULL,
    .max_capacity_gb = 50,
    .show_help = 0,
};

#define OPTION(t, p) { t, offsetof(struct cache_options, p), 1 }
static const struct fuse_opt option_spec[] = {
    OPTION("--origin=%s", origin),
    OPTION("--cache-dir=%s", cache_dir),
    OPTION("--temp-dir=%s", temp_dir),
    OPTION("--locks-dir=%s", locks_dir),
    OPTION("--db=%s", db_path),
    OPTION("--dataset-id=%s", dataset_id),
    OPTION("--dataset-version=%s", dataset_version),
    OPTION("--max-capacity-gb=%d", max_capacity_gb),
    OPTION("-h", show_help),
    OPTION("--help", show_help),
    FUSE_OPT_END
};

static double get_time_sec(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + (tv.tv_usec / 1000000.0);
}

static void ensure_parent_dirs(const char *filepath) {
    char temp[1024];
    snprintf(temp, sizeof(temp), "%s", filepath);
    char *p = temp;
    if (*p == '/') p++;
    while ((p = strchr(p, '/')) != NULL) {
        *p = '\0';
        mkdir(temp, 0755);
        *p = '/';
        p++;
    }
}

static void sanitize_lock_name(const char *rel_path, char *out, size_t maxlen) {
    size_t i;
    for (i = 0; i < maxlen - 1 && rel_path[i] != '\0'; i++) {
        char c = rel_path[i];
        if (c == '/' || c == '\\' || c == ':') {
            out[i] = '_';
        } else {
            out[i] = c;
        }
    }
    out[i] = '\0';
}

static sqlite3 *open_db(void) {
    sqlite3 *db = NULL;
    int rc = sqlite3_open_v2(
        options.db_path,
        &db,
        SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX,
        NULL
    );
    if (rc != SQLITE_OK) {
        if (db) sqlite3_close(db);
        return NULL;
    }
    sqlite3_busy_timeout(db, 5000);
    sqlite3_exec(db, "PRAGMA journal_mode = WAL;", NULL, NULL, NULL);
    
    const char *schema =
        "CREATE TABLE IF NOT EXISTS cached_files ("
        "  dataset_id TEXT NOT NULL,"
        "  dataset_version TEXT NOT NULL,"
        "  relative_path TEXT NOT NULL,"
        "  file_size INTEGER NOT NULL,"
        "  sha256 TEXT NOT NULL,"
        "  local_path TEXT NOT NULL,"
        "  valid INTEGER NOT NULL,"
        "  last_access_time REAL,"
        "  access_count INTEGER DEFAULT 0,"
        "  download_time REAL,"
        "  PRIMARY KEY (dataset_id, dataset_version, relative_path)"
        ");"
        "CREATE TABLE IF NOT EXISTS cache_statistics ("
        "  id INTEGER PRIMARY KEY CHECK (id = 1),"
        "  cache_hits INTEGER DEFAULT 0,"
        "  cache_misses INTEGER DEFAULT 0,"
        "  downloaded_bytes INTEGER DEFAULT 0,"
        "  download_failures INTEGER DEFAULT 0,"
        "  cached_bytes INTEGER DEFAULT 0,"
        "  cached_files INTEGER DEFAULT 0,"
        "  registry_revision INTEGER DEFAULT 0"
        ");"
        "INSERT OR IGNORE INTO cache_statistics (id, cache_hits, cache_misses, downloaded_bytes, download_failures, cached_bytes, cached_files, registry_revision)"
        "VALUES (1, 0, 0, 0, 0, 0, 0, 0);";
    sqlite3_exec(db, schema, NULL, NULL, NULL);

    return db;
}

static int is_file_valid_in_db(const char *rel_path) {
    sqlite3 *db = open_db();
    if (!db) return 0;

    const char *sql = "SELECT valid FROM cached_files WHERE dataset_id = ? AND dataset_version = ? AND relative_path = ? AND valid = 1;";
    sqlite3_stmt *stmt = NULL;
    int valid = 0;

    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, options.dataset_id, -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 2, options.dataset_version, -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 3, rel_path, -1, SQLITE_STATIC);
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            valid = sqlite3_column_int(stmt, 0);
        }
        sqlite3_finalize(stmt);
    }
    sqlite3_close(db);
    return valid;
}

static void record_hit_in_db(const char *rel_path) {
    sqlite3 *db = open_db();
    if (!db) return;

    double now = get_time_sec();
    const char *sql_update =
        "UPDATE cached_files SET last_access_time = ?, access_count = access_count + 1 "
        "WHERE dataset_id = ? AND dataset_version = ? AND relative_path = ?;";
    sqlite3_stmt *stmt = NULL;
    if (sqlite3_prepare_v2(db, sql_update, -1, &stmt, NULL) == SQLITE_OK) {
        sqlite3_bind_double(stmt, 1, now);
        sqlite3_bind_text(stmt, 2, options.dataset_id, -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 3, options.dataset_version, -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 4, rel_path, -1, SQLITE_STATIC);
        sqlite3_step(stmt);
        sqlite3_finalize(stmt);
    }
    sqlite3_exec(db, "UPDATE cache_statistics SET cache_hits = cache_hits + 1 WHERE id = 1;", NULL, NULL, NULL);
    sqlite3_close(db);
}

static void record_commit_in_db(
    const char *rel_path,
    off_t file_size,
    const char *sha256_hex,
    const char *local_path,
    double download_time
) {
    sqlite3 *db = open_db();
    if (!db) return;

    double now = get_time_sec();
    const char *sql_insert =
        "INSERT INTO cached_files ("
        "  dataset_id, dataset_version, relative_path, file_size, sha256, "
        "  local_path, valid, last_access_time, access_count, download_time"
        ") VALUES (?, ?, ?, ?, ?, ?, 1, ?, 1, ?) "
        "ON CONFLICT(dataset_id, dataset_version, relative_path) DO UPDATE SET "
        "  file_size = excluded.file_size, "
        "  sha256 = excluded.sha256, "
        "  local_path = excluded.local_path, "
        "  valid = 1, "
        "  last_access_time = excluded.last_access_time, "
        "  access_count = access_count + 1, "
        "  download_time = excluded.download_time;";

    sqlite3_stmt *stmt = NULL;
    if (sqlite3_prepare_v2(db, sql_insert, -1, &stmt, NULL) == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, options.dataset_id, -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 2, options.dataset_version, -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 3, rel_path, -1, SQLITE_STATIC);
        sqlite3_bind_int64(stmt, 4, (sqlite3_int64)file_size);
        sqlite3_bind_text(stmt, 5, sha256_hex, -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 6, local_path, -1, SQLITE_STATIC);
        sqlite3_bind_double(stmt, 7, now);
        sqlite3_bind_double(stmt, 8, download_time);
        sqlite3_step(stmt);
        sqlite3_finalize(stmt);
    }

    const char *sql_stats =
        "UPDATE cache_statistics SET "
        "  cache_misses = cache_misses + 1, "
        "  downloaded_bytes = downloaded_bytes + ?, "
        "  cached_bytes = (SELECT COALESCE(SUM(file_size), 0) FROM cached_files WHERE valid = 1), "
        "  cached_files = (SELECT COUNT(*) FROM cached_files WHERE valid = 1), "
        "  registry_revision = registry_revision + 1 "
        "WHERE id = 1;";

    if (sqlite3_prepare_v2(db, sql_stats, -1, &stmt, NULL) == SQLITE_OK) {
        sqlite3_bind_int64(stmt, 1, (sqlite3_int64)file_size);
        sqlite3_step(stmt);
        sqlite3_finalize(stmt);
    }

    sqlite3_close(db);
}

static void make_path(char *dest, size_t maxlen, const char *base, const char *path) {
    if (path[0] == '/') path++;
    snprintf(dest, maxlen, "%s/%s", base, path);
}

/* FUSE Operations */

static int optimus_getattr(const char *path, struct stat *stbuf, struct fuse_file_info *fi) {
    (void) fi;
    char full_path[1024];

    if (strcmp(path, "/") == 0) {
        stbuf->st_mode = S_IFDIR | 0755;
        stbuf->st_nlink = 2;
        return 0;
    }

    const char *rel_path = path[0] == '/' ? path + 1 : path;

    // First check local cache
    make_path(full_path, sizeof(full_path), options.cache_dir, rel_path);
    if (lstat(full_path, stbuf) == 0) {
        return 0;
    }

    // Otherwise check origin
    make_path(full_path, sizeof(full_path), options.origin, rel_path);
    if (lstat(full_path, stbuf) == 0) {
        return 0;
    }

    return -ENOENT;
}

static int optimus_readdir(
    const char *path,
    void *buf,
    fuse_fill_dir_t filler,
    off_t offset,
    struct fuse_file_info *fi,
    enum fuse_readdir_flags flags
) {
    (void) offset;
    (void) fi;
    (void) flags;

    char full_origin[1024];
    const char *rel_path = path[0] == '/' ? path + 1 : path;
    make_path(full_origin, sizeof(full_origin), options.origin, rel_path);

    DIR *dp = opendir(full_origin);
    if (!dp) return -errno;

    filler(buf, ".", NULL, 0, 0);
    filler(buf, "..", NULL, 0, 0);

    struct dirent *de;
    while ((de = readdir(dp)) != NULL) {
        if (strcmp(de->d_name, ".") == 0 || strcmp(de->d_name, "..") == 0)
            continue;
        if (strcmp(de->d_name, "manifest.json") == 0)
            continue;
        filler(buf, de->d_name, NULL, 0, 0);
    }
    closedir(dp);
    return 0;
}

static int optimus_open(const char *path, struct fuse_file_info *fi) {
    const char *rel_path = path[0] == '/' ? path + 1 : path;
    char local_path[1024];
    char origin_path[1024];
    char lock_path[1024];
    char temp_path[1024];
    char sanitized_lock[512];

    make_path(local_path, sizeof(local_path), options.cache_dir, rel_path);
    make_path(origin_path, sizeof(origin_path), options.origin, rel_path);

    // 1. Check if already in cache and valid
    struct stat st;
    if (stat(local_path, &st) == 0 && is_file_valid_in_db(rel_path)) {
        int fd = open(local_path, O_RDONLY);
        if (fd < 0) return -errno;
        fi->fh = fd;
        record_hit_in_db(rel_path);
        return 0;
    }

    // 2. Cache Miss: acquire process-shared file lock
    sanitize_lock_name(rel_path, sanitized_lock, sizeof(sanitized_lock));
    snprintf(lock_path, sizeof(lock_path), "%s/%s.lock", options.locks_dir, sanitized_lock);
    ensure_parent_dirs(lock_path);

    int lock_fd = open(lock_path, O_CREAT | O_RDWR, 0666);
    if (lock_fd < 0) return -errno;

    if (flock(lock_fd, LOCK_EX) < 0) {
        close(lock_fd);
        return -errno;
    }

    // Re-check cache after lock acquisition
    if (stat(local_path, &st) == 0 && is_file_valid_in_db(rel_path)) {
        flock(lock_fd, LOCK_UN);
        close(lock_fd);
        int fd = open(local_path, O_RDONLY);
        if (fd < 0) return -errno;
        fi->fh = fd;
        record_hit_in_db(rel_path);
        return 0;
    }

    // 3. Open origin file over NFS
    int src_fd = open(origin_path, O_RDONLY);
    if (src_fd < 0) {
        flock(lock_fd, LOCK_UN);
        close(lock_fd);
        return -errno;
    }

    // 4. Create unique temporary download file
    double start_t = get_time_sec();
    snprintf(
        temp_path,
        sizeof(temp_path),
        "%s/%s.%d.%ld.downloading",
        options.temp_dir,
        sanitized_lock,
        getpid(),
        random()
    );
    ensure_parent_dirs(temp_path);

    int dst_fd = open(temp_path, O_CREAT | O_WRONLY | O_TRUNC, 0644);
    if (dst_fd < 0) {
        close(src_fd);
        flock(lock_fd, LOCK_UN);
        close(lock_fd);
        return -errno;
    }

    // 5. Stream copy while computing SHA-256
    SHA256_CTX sha_ctx;
    SHA256_Init(&sha_ctx);

    char buffer[CHUNK_SIZE];
    ssize_t bytes_read;
    off_t total_bytes = 0;
    int copy_failed = 0;

    while ((bytes_read = read(src_fd, buffer, sizeof(buffer))) > 0) {
        ssize_t written = 0;
        while (written < bytes_read) {
            ssize_t w = write(dst_fd, buffer + written, bytes_read - written);
            if (w <= 0) {
                copy_failed = 1;
                break;
            }
            written += w;
        }
        if (copy_failed) break;
        SHA256_Update(&sha_ctx, buffer, bytes_read);
        total_bytes += bytes_read;
    }

    if (bytes_read < 0 || copy_failed) {
        close(src_fd);
        close(dst_fd);
        unlink(temp_path);
        flock(lock_fd, LOCK_UN);
        close(lock_fd);
        return -EIO;
    }

    unsigned char digest[SHA256_DIGEST_LENGTH];
    SHA256_Final(digest, &sha_ctx);

    char sha256_hex[SHA256_DIGEST_LENGTH * 2 + 1];
    for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
        sprintf(sha256_hex + (i * 2), "%02x", digest[i]);
    }
    sha256_hex[64] = '\0';

    // Flush and synchronize temp file
    fsync(dst_fd);
    close(dst_fd);
    close(src_fd);

    // 6. Atomically rename to final cache destination
    ensure_parent_dirs(local_path);
    if (rename(temp_path, local_path) != 0) {
        unlink(temp_path);
        flock(lock_fd, LOCK_UN);
        close(lock_fd);
        return -errno;
    }

    // 7. Commit metadata to SQLite
    double duration = get_time_sec() - start_t;
    record_commit_in_db(rel_path, total_bytes, sha256_hex, local_path, duration);

    // 8. Release lock
    flock(lock_fd, LOCK_UN);
    close(lock_fd);

    // 9. Open and serve the newly cached local file
    int final_fd = open(local_path, O_RDONLY);
    if (final_fd < 0) return -errno;

    fi->fh = final_fd;
    return 0;
}

static int optimus_read(
    const char *path,
    char *buf,
    size_t size,
    off_t offset,
    struct fuse_file_info *fi
) {
    (void) path;
    ssize_t res = pread(fi->fh, buf, size, offset);
    if (res < 0) return -errno;
    return res;
}

static int optimus_release(const char *path, struct fuse_file_info *fi) {
    (void) path;
    if (fi->fh > 0) {
        close(fi->fh);
    }
    return 0;
}

static int optimus_statfs(const char *path, struct statvfs *stbuf) {
    (void) path;
    if (statvfs(options.cache_dir, stbuf) < 0) {
        return -errno;
    }
    return 0;
}

static int optimus_access(const char *path, int mask) {
    if (mask & W_OK) return -EROFS;
    char full_path[1024];
    const char *rel_path = path[0] == '/' ? path + 1 : path;

    make_path(full_path, sizeof(full_path), options.cache_dir, rel_path);
    if (access(full_path, mask) == 0) return 0;

    make_path(full_path, sizeof(full_path), options.origin, rel_path);
    if (access(full_path, mask) == 0) return 0;

    return -ENOENT;
}

/* Read-only rejection handlers */
static int optimus_write(const char *path, const char *buf, size_t size, off_t offset, struct fuse_file_info *fi) {
    (void)path; (void)buf; (void)size; (void)offset; (void)fi;
    return -EROFS;
}

static int optimus_create(const char *path, mode_t mode, struct fuse_file_info *fi) {
    (void)path; (void)mode; (void)fi;
    return -EROFS;
}

static int optimus_unlink(const char *path) { (void)path; return -EROFS; }
static int optimus_mkdir(const char *path, mode_t mode) { (void)path; (void)mode; return -EROFS; }
static int optimus_rmdir(const char *path) { (void)path; return -EROFS; }

static const struct fuse_operations optimus_oper = {
    .getattr = optimus_getattr,
    .readdir = optimus_readdir,
    .open = optimus_open,
    .read = optimus_read,
    .release = optimus_release,
    .statfs = optimus_statfs,
    .access = optimus_access,
    .write = optimus_write,
    .create = optimus_create,
    .unlink = optimus_unlink,
    .mkdir = optimus_mkdir,
    .rmdir = optimus_rmdir,
};

int main(int argc, char *argv[]) {
    struct fuse_args args = FUSE_ARGS_INIT(argc, argv);

    if (fuse_opt_parse(&args, &options, option_spec, NULL) == -1) {
        return 1;
    }

    if (!options.cache_dir) options.cache_dir = strdup("/var/cache/optimus/files");
    if (!options.temp_dir) options.temp_dir = strdup("/var/cache/optimus/temporary");
    if (!options.locks_dir) options.locks_dir = strdup("/var/cache/optimus/locks");
    if (!options.db_path) options.db_path = strdup("/var/cache/optimus/cache.db");
    if (!options.dataset_id) options.dataset_id = strdup("default");
    if (!options.dataset_version) options.dataset_version = strdup("v1");

    if (options.show_help || !options.origin) {
        fprintf(stderr, "Usage: fuse_cache [options] <mountpoint>\n\n");
        fprintf(stderr, "Options:\n");
        fprintf(stderr, "  --origin=<dir>             NFS origin directory (required)\n");
        fprintf(stderr, "  --cache-dir=<dir>          Local cache storage directory\n");
        fprintf(stderr, "  --temp-dir=<dir>           Temporary download directory\n");
        fprintf(stderr, "  --locks-dir=<dir>          Process locks directory\n");
        fprintf(stderr, "  --db=<path>                SQLite cache database path\n");
        fprintf(stderr, "  --dataset-id=<id>          Dataset ID\n");
        fprintf(stderr, "  --dataset-version=<ver>    Dataset Version\n");
        fprintf(stderr, "  --max-capacity-gb=<num>    Maximum cache capacity in GB\n");
        return 1;
    }

    // Ensure directories exist
    ensure_parent_dirs(options.cache_dir);
    ensure_parent_dirs(options.temp_dir);
    ensure_parent_dirs(options.locks_dir);

    return fuse_main(args.argc, args.argv, &optimus_oper, NULL);
}
