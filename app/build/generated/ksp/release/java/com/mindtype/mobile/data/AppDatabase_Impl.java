package com.mindtype.mobile.data;

import androidx.annotation.NonNull;
import androidx.room.DatabaseConfiguration;
import androidx.room.InvalidationTracker;
import androidx.room.RoomDatabase;
import androidx.room.RoomOpenHelper;
import androidx.room.migration.AutoMigrationSpec;
import androidx.room.migration.Migration;
import androidx.room.util.DBUtil;
import androidx.room.util.TableInfo;
import androidx.sqlite.db.SupportSQLiteDatabase;
import androidx.sqlite.db.SupportSQLiteOpenHelper;
import com.mindtype.mobile.data.dao.FeatureWindowDao;
import com.mindtype.mobile.data.dao.FeatureWindowDao_Impl;
import com.mindtype.mobile.data.dao.KeystrokeEventDao;
import com.mindtype.mobile.data.dao.KeystrokeEventDao_Impl;
import com.mindtype.mobile.data.dao.SessionDao;
import com.mindtype.mobile.data.dao.SessionDao_Impl;
import com.mindtype.mobile.data.dao.StressLabelDao;
import com.mindtype.mobile.data.dao.StressLabelDao_Impl;
import com.mindtype.mobile.data.dao.UserDao;
import com.mindtype.mobile.data.dao.UserDao_Impl;
import java.lang.Class;
import java.lang.Override;
import java.lang.String;
import java.lang.SuppressWarnings;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import javax.annotation.processing.Generated;

@Generated("androidx.room.RoomProcessor")
@SuppressWarnings({"unchecked", "deprecation"})
public final class AppDatabase_Impl extends AppDatabase {
  private volatile UserDao _userDao;

  private volatile SessionDao _sessionDao;

  private volatile KeystrokeEventDao _keystrokeEventDao;

  private volatile FeatureWindowDao _featureWindowDao;

  private volatile StressLabelDao _stressLabelDao;

  @Override
  @NonNull
  protected SupportSQLiteOpenHelper createOpenHelper(@NonNull final DatabaseConfiguration config) {
    final SupportSQLiteOpenHelper.Callback _openCallback = new RoomOpenHelper(config, new RoomOpenHelper.Delegate(1) {
      @Override
      public void createAllTables(@NonNull final SupportSQLiteDatabase db) {
        db.execSQL("CREATE TABLE IF NOT EXISTS `users` (`user_id` TEXT NOT NULL, `created_at` INTEGER NOT NULL, PRIMARY KEY(`user_id`))");
        db.execSQL("CREATE TABLE IF NOT EXISTS `sessions` (`session_id` TEXT NOT NULL, `user_id` TEXT NOT NULL, `start_time` INTEGER NOT NULL, `end_time` INTEGER, PRIMARY KEY(`session_id`), FOREIGN KEY(`user_id`) REFERENCES `users`(`user_id`) ON UPDATE NO ACTION ON DELETE CASCADE )");
        db.execSQL("CREATE TABLE IF NOT EXISTS `keystroke_events` (`event_id` INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, `session_id` TEXT NOT NULL, `user_id` TEXT NOT NULL, `timestamp` INTEGER NOT NULL, `key_code` INTEGER NOT NULL, `down_time` INTEGER NOT NULL, `event_time` INTEGER NOT NULL, `dwell_time` REAL NOT NULL, `flight_time` REAL NOT NULL, `touch_pressure` REAL NOT NULL, `touch_size` REAL NOT NULL, `is_backspace` INTEGER NOT NULL, FOREIGN KEY(`session_id`) REFERENCES `sessions`(`session_id`) ON UPDATE NO ACTION ON DELETE CASCADE , FOREIGN KEY(`user_id`) REFERENCES `users`(`user_id`) ON UPDATE NO ACTION ON DELETE CASCADE )");
        db.execSQL("CREATE TABLE IF NOT EXISTS `feature_windows` (`window_id` INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, `session_id` TEXT NOT NULL, `window_start` INTEGER NOT NULL, `window_end` INTEGER NOT NULL, `mean_dwell` REAL NOT NULL, `std_dwell` REAL NOT NULL, `mean_flight` REAL NOT NULL, `std_flight` REAL NOT NULL, `typing_speed` REAL NOT NULL, `backspace_rate` REAL NOT NULL, `pause_count` INTEGER NOT NULL, `mean_pressure` REAL NOT NULL, `gyro_std` REAL NOT NULL, `predicted_class` TEXT NOT NULL, `stress_label` TEXT, FOREIGN KEY(`session_id`) REFERENCES `sessions`(`session_id`) ON UPDATE NO ACTION ON DELETE CASCADE )");
        db.execSQL("CREATE TABLE IF NOT EXISTS `stress_labels` (`label_id` INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, `session_id` TEXT NOT NULL, `timestamp` INTEGER NOT NULL, `raw_score` INTEGER NOT NULL, `mapped_class` TEXT NOT NULL, FOREIGN KEY(`session_id`) REFERENCES `sessions`(`session_id`) ON UPDATE NO ACTION ON DELETE CASCADE )");
        db.execSQL("CREATE TABLE IF NOT EXISTS room_master_table (id INTEGER PRIMARY KEY,identity_hash TEXT)");
        db.execSQL("INSERT OR REPLACE INTO room_master_table (id,identity_hash) VALUES(42, '0f122065813dcf8942cb5843df11ea97')");
      }

      @Override
      public void dropAllTables(@NonNull final SupportSQLiteDatabase db) {
        db.execSQL("DROP TABLE IF EXISTS `users`");
        db.execSQL("DROP TABLE IF EXISTS `sessions`");
        db.execSQL("DROP TABLE IF EXISTS `keystroke_events`");
        db.execSQL("DROP TABLE IF EXISTS `feature_windows`");
        db.execSQL("DROP TABLE IF EXISTS `stress_labels`");
        final List<? extends RoomDatabase.Callback> _callbacks = mCallbacks;
        if (_callbacks != null) {
          for (RoomDatabase.Callback _callback : _callbacks) {
            _callback.onDestructiveMigration(db);
          }
        }
      }

      @Override
      public void onCreate(@NonNull final SupportSQLiteDatabase db) {
        final List<? extends RoomDatabase.Callback> _callbacks = mCallbacks;
        if (_callbacks != null) {
          for (RoomDatabase.Callback _callback : _callbacks) {
            _callback.onCreate(db);
          }
        }
      }

      @Override
      public void onOpen(@NonNull final SupportSQLiteDatabase db) {
        mDatabase = db;
        db.execSQL("PRAGMA foreign_keys = ON");
        internalInitInvalidationTracker(db);
        final List<? extends RoomDatabase.Callback> _callbacks = mCallbacks;
        if (_callbacks != null) {
          for (RoomDatabase.Callback _callback : _callbacks) {
            _callback.onOpen(db);
          }
        }
      }

      @Override
      public void onPreMigrate(@NonNull final SupportSQLiteDatabase db) {
        DBUtil.dropFtsSyncTriggers(db);
      }

      @Override
      public void onPostMigrate(@NonNull final SupportSQLiteDatabase db) {
      }

      @Override
      @NonNull
      public RoomOpenHelper.ValidationResult onValidateSchema(
          @NonNull final SupportSQLiteDatabase db) {
        final HashMap<String, TableInfo.Column> _columnsUsers = new HashMap<String, TableInfo.Column>(2);
        _columnsUsers.put("user_id", new TableInfo.Column("user_id", "TEXT", true, 1, null, TableInfo.CREATED_FROM_ENTITY));
        _columnsUsers.put("created_at", new TableInfo.Column("created_at", "INTEGER", true, 0, null, TableInfo.CREATED_FROM_ENTITY));
        final HashSet<TableInfo.ForeignKey> _foreignKeysUsers = new HashSet<TableInfo.ForeignKey>(0);
        final HashSet<TableInfo.Index> _indicesUsers = new HashSet<TableInfo.Index>(0);
        final TableInfo _infoUsers = new TableInfo("users", _columnsUsers, _foreignKeysUsers, _indicesUsers);
        final TableInfo _existingUsers = TableInfo.read(db, "users");
        if (!_infoUsers.equals(_existingUsers)) {
          return new RoomOpenHelper.ValidationResult(false, "users(com.mindtype.mobile.data.entity.UserEntity).\n"
                  + " Expected:\n" + _infoUsers + "\n"
                  + " Found:\n" + _existingUsers);
        }
        final HashMap<String, TableInfo.Column> _columnsSessions = new HashMap<String, TableInfo.Column>(4);
        _columnsSessions.put("session_id", new TableInfo.Column("session_id", "TEXT", true, 1, null, TableInfo.CREATED_FROM_ENTITY));
        _columnsSessions.put("user_id", new TableInfo.Column("user_id", "TEXT", true, 0, null, TableInfo.CREATED_FROM_ENTITY));
        _columnsSessions.put("start_time", new TableInfo.Column("start_time", "INTEGER", true, 0, null, TableInfo.CREATED_FROM_ENTITY));
        _columnsSessions.put("end_time", new TableInfo.Column("end_time", "INTEGER", false, 0, null, TableInfo.CREATED_FROM_ENTITY));
        final HashSet<TableInfo.ForeignKey> _foreignKeysSessions = new HashSet<TableInfo.ForeignKey>(1);
        _foreignKeysSessions.add(new TableInfo.ForeignKey("users", "CASCADE", "NO ACTION", Arrays.asList("user_id"), Arrays.asList("user_id")));
        final HashSet<TableInfo.Index> _indicesSessions = new HashSet<TableInfo.Index>(0);
        final TableInfo _infoSessions = new TableInfo("sessions", _columnsSessions, _foreignKeysSessions, _indicesSessions);
        final TableInfo _existingSessions = TableInfo.read(db, "sessions");
        if (!_infoSessions.equals(_existingSessions)) {
          return new RoomOpenHelper.ValidationResult(false, "sessions(com.mindtype.mobile.data.entity.SessionEntity).\n"
                  + " Expected:\n" + _infoSessions + "\n"
                  + " Found:\n" + _existingSessions);
        }
        final HashMap<String, TableInfo.Column> _columnsKeystrokeEvents = new HashMap<String, TableInfo.Column>(12);
        _columnsKeystrokeEvents.put("event_id", new TableInfo.Column("event_id", "INTEGER", true, 1, null, TableInfo.CREATED_FROM_ENTITY));
        _columnsKeystrokeEvents.put("session_id", new TableInfo.Column("session_id", "TEXT", true, 0, null, TableInfo.CREATED_FROM_ENTITY));
        _columnsKeystrokeEvents.put("user_id", new TableInfo.Column("user_id", "TEXT", true, 0, null, TableInfo.CREATED_FROM_ENTITY));
        _columnsKeystrokeEvents.put("timestamp", new TableInfo.Column("timestamp", "INTEGER", true, 0, null, TableInfo.CREATED_FROM_ENTITY));
        _columnsKeystrokeEvents.put("key_code", new TableInfo.Column("key_code", "INTEGER", true, 0, null, TableInfo.CREATED_FROM_ENTITY));
        _columnsKeystrokeEvents.put("down_time", new TableInfo.Column("down_time", "INTEGER", true, 0, null, TableInfo.CREATED_FROM_ENTITY));
        _columnsKeystrokeEvents.put("event_time", new TableInfo.Column("event_time", "INTEGER", true, 0, null, TableInfo.CREATED_FROM_ENTITY));
        _columnsKeystrokeEvents.put("dwell_time", new TableInfo.Column("dwell_time", "REAL", true, 0, null, TableInfo.CREATED_FROM_ENTITY));
        _columnsKeystrokeEvents.put("flight_time", new TableInfo.Column("flight_time", "REAL", true, 0, null, TableInfo.CREATED_FROM_ENTITY));
        _columnsKeystrokeEvents.put("touch_pressure", new TableInfo.Column("touch_pressure", "REAL", true, 0, null, TableInfo.CREATED_FROM_ENTITY));
        _columnsKeystrokeEvents.put("touch_size", new TableInfo.Column("touch_size", "REAL", true, 0, null, TableInfo.CREATED_FROM_ENTITY));
        _columnsKeystrokeEvents.put("is_backspace", new TableInfo.Column("is_backspace", "INTEGER", true, 0, null, TableInfo.CREATED_FROM_ENTITY));
        final HashSet<TableInfo.ForeignKey> _foreignKeysKeystrokeEvents = new HashSet<TableInfo.ForeignKey>(2);
        _foreignKeysKeystrokeEvents.add(new TableInfo.ForeignKey("sessions", "CASCADE", "NO ACTION", Arrays.asList("session_id"), Arrays.asList("session_id")));
        _foreignKeysKeystrokeEvents.add(new TableInfo.ForeignKey("users", "CASCADE", "NO ACTION", Arrays.asList("user_id"), Arrays.asList("user_id")));
        final HashSet<TableInfo.Index> _indicesKeystrokeEvents = new HashSet<TableInfo.Index>(0);
        final TableInfo _infoKeystrokeEvents = new TableInfo("keystroke_events", _columnsKeystrokeEvents, _foreignKeysKeystrokeEvents, _indicesKeystrokeEvents);
        final TableInfo _existingKeystrokeEvents = TableInfo.read(db, "keystroke_events");
        if (!_infoKeystrokeEvents.equals(_existingKeystrokeEvents)) {
          return new RoomOpenHelper.ValidationResult(false, "keystroke_events(com.mindtype.mobile.data.entity.KeystrokeEventEntity).\n"
                  + " Expected:\n" + _infoKeystrokeEvents + "\n"
                  + " Found:\n" + _existingKeystrokeEvents);
        }
        final HashMap<String, TableInfo.Column> _columnsFeatureWindows = new HashMap<String, TableInfo.Column>(15);
        _columnsFeatureWindows.put("window_id", new TableInfo.Column("window_id", "INTEGER", true, 1, null, TableInfo.CREATED_FROM_ENTITY));
        _columnsFeatureWindows.put("session_id", new TableInfo.Column("session_id", "TEXT", true, 0, null, TableInfo.CREATED_FROM_ENTITY));
        _columnsFeatureWindows.put("window_start", new TableInfo.Column("window_start", "INTEGER", true, 0, null, TableInfo.CREATED_FROM_ENTITY));
        _columnsFeatureWindows.put("window_end", new TableInfo.Column("window_end", "INTEGER", true, 0, null, TableInfo.CREATED_FROM_ENTITY));
        _columnsFeatureWindows.put("mean_dwell", new TableInfo.Column("mean_dwell", "REAL", true, 0, null, TableInfo.CREATED_FROM_ENTITY));
        _columnsFeatureWindows.put("std_dwell", new TableInfo.Column("std_dwell", "REAL", true, 0, null, TableInfo.CREATED_FROM_ENTITY));
        _columnsFeatureWindows.put("mean_flight", new TableInfo.Column("mean_flight", "REAL", true, 0, null, TableInfo.CREATED_FROM_ENTITY));
        _columnsFeatureWindows.put("std_flight", new TableInfo.Column("std_flight", "REAL", true, 0, null, TableInfo.CREATED_FROM_ENTITY));
        _columnsFeatureWindows.put("typing_speed", new TableInfo.Column("typing_speed", "REAL", true, 0, null, TableInfo.CREATED_FROM_ENTITY));
        _columnsFeatureWindows.put("backspace_rate", new TableInfo.Column("backspace_rate", "REAL", true, 0, null, TableInfo.CREATED_FROM_ENTITY));
        _columnsFeatureWindows.put("pause_count", new TableInfo.Column("pause_count", "INTEGER", true, 0, null, TableInfo.CREATED_FROM_ENTITY));
        _columnsFeatureWindows.put("mean_pressure", new TableInfo.Column("mean_pressure", "REAL", true, 0, null, TableInfo.CREATED_FROM_ENTITY));
        _columnsFeatureWindows.put("gyro_std", new TableInfo.Column("gyro_std", "REAL", true, 0, null, TableInfo.CREATED_FROM_ENTITY));
        _columnsFeatureWindows.put("predicted_class", new TableInfo.Column("predicted_class", "TEXT", true, 0, null, TableInfo.CREATED_FROM_ENTITY));
        _columnsFeatureWindows.put("stress_label", new TableInfo.Column("stress_label", "TEXT", false, 0, null, TableInfo.CREATED_FROM_ENTITY));
        final HashSet<TableInfo.ForeignKey> _foreignKeysFeatureWindows = new HashSet<TableInfo.ForeignKey>(1);
        _foreignKeysFeatureWindows.add(new TableInfo.ForeignKey("sessions", "CASCADE", "NO ACTION", Arrays.asList("session_id"), Arrays.asList("session_id")));
        final HashSet<TableInfo.Index> _indicesFeatureWindows = new HashSet<TableInfo.Index>(0);
        final TableInfo _infoFeatureWindows = new TableInfo("feature_windows", _columnsFeatureWindows, _foreignKeysFeatureWindows, _indicesFeatureWindows);
        final TableInfo _existingFeatureWindows = TableInfo.read(db, "feature_windows");
        if (!_infoFeatureWindows.equals(_existingFeatureWindows)) {
          return new RoomOpenHelper.ValidationResult(false, "feature_windows(com.mindtype.mobile.data.entity.FeatureWindowEntity).\n"
                  + " Expected:\n" + _infoFeatureWindows + "\n"
                  + " Found:\n" + _existingFeatureWindows);
        }
        final HashMap<String, TableInfo.Column> _columnsStressLabels = new HashMap<String, TableInfo.Column>(5);
        _columnsStressLabels.put("label_id", new TableInfo.Column("label_id", "INTEGER", true, 1, null, TableInfo.CREATED_FROM_ENTITY));
        _columnsStressLabels.put("session_id", new TableInfo.Column("session_id", "TEXT", true, 0, null, TableInfo.CREATED_FROM_ENTITY));
        _columnsStressLabels.put("timestamp", new TableInfo.Column("timestamp", "INTEGER", true, 0, null, TableInfo.CREATED_FROM_ENTITY));
        _columnsStressLabels.put("raw_score", new TableInfo.Column("raw_score", "INTEGER", true, 0, null, TableInfo.CREATED_FROM_ENTITY));
        _columnsStressLabels.put("mapped_class", new TableInfo.Column("mapped_class", "TEXT", true, 0, null, TableInfo.CREATED_FROM_ENTITY));
        final HashSet<TableInfo.ForeignKey> _foreignKeysStressLabels = new HashSet<TableInfo.ForeignKey>(1);
        _foreignKeysStressLabels.add(new TableInfo.ForeignKey("sessions", "CASCADE", "NO ACTION", Arrays.asList("session_id"), Arrays.asList("session_id")));
        final HashSet<TableInfo.Index> _indicesStressLabels = new HashSet<TableInfo.Index>(0);
        final TableInfo _infoStressLabels = new TableInfo("stress_labels", _columnsStressLabels, _foreignKeysStressLabels, _indicesStressLabels);
        final TableInfo _existingStressLabels = TableInfo.read(db, "stress_labels");
        if (!_infoStressLabels.equals(_existingStressLabels)) {
          return new RoomOpenHelper.ValidationResult(false, "stress_labels(com.mindtype.mobile.data.entity.StressLabelEntity).\n"
                  + " Expected:\n" + _infoStressLabels + "\n"
                  + " Found:\n" + _existingStressLabels);
        }
        return new RoomOpenHelper.ValidationResult(true, null);
      }
    }, "0f122065813dcf8942cb5843df11ea97", "a67d7c7a8c0c5ac90a3d346c85663527");
    final SupportSQLiteOpenHelper.Configuration _sqliteConfig = SupportSQLiteOpenHelper.Configuration.builder(config.context).name(config.name).callback(_openCallback).build();
    final SupportSQLiteOpenHelper _helper = config.sqliteOpenHelperFactory.create(_sqliteConfig);
    return _helper;
  }

  @Override
  @NonNull
  protected InvalidationTracker createInvalidationTracker() {
    final HashMap<String, String> _shadowTablesMap = new HashMap<String, String>(0);
    final HashMap<String, Set<String>> _viewTables = new HashMap<String, Set<String>>(0);
    return new InvalidationTracker(this, _shadowTablesMap, _viewTables, "users","sessions","keystroke_events","feature_windows","stress_labels");
  }

  @Override
  public void clearAllTables() {
    super.assertNotMainThread();
    final SupportSQLiteDatabase _db = super.getOpenHelper().getWritableDatabase();
    final boolean _supportsDeferForeignKeys = android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.LOLLIPOP;
    try {
      if (!_supportsDeferForeignKeys) {
        _db.execSQL("PRAGMA foreign_keys = FALSE");
      }
      super.beginTransaction();
      if (_supportsDeferForeignKeys) {
        _db.execSQL("PRAGMA defer_foreign_keys = TRUE");
      }
      _db.execSQL("DELETE FROM `users`");
      _db.execSQL("DELETE FROM `sessions`");
      _db.execSQL("DELETE FROM `keystroke_events`");
      _db.execSQL("DELETE FROM `feature_windows`");
      _db.execSQL("DELETE FROM `stress_labels`");
      super.setTransactionSuccessful();
    } finally {
      super.endTransaction();
      if (!_supportsDeferForeignKeys) {
        _db.execSQL("PRAGMA foreign_keys = TRUE");
      }
      _db.query("PRAGMA wal_checkpoint(FULL)").close();
      if (!_db.inTransaction()) {
        _db.execSQL("VACUUM");
      }
    }
  }

  @Override
  @NonNull
  protected Map<Class<?>, List<Class<?>>> getRequiredTypeConverters() {
    final HashMap<Class<?>, List<Class<?>>> _typeConvertersMap = new HashMap<Class<?>, List<Class<?>>>();
    _typeConvertersMap.put(UserDao.class, UserDao_Impl.getRequiredConverters());
    _typeConvertersMap.put(SessionDao.class, SessionDao_Impl.getRequiredConverters());
    _typeConvertersMap.put(KeystrokeEventDao.class, KeystrokeEventDao_Impl.getRequiredConverters());
    _typeConvertersMap.put(FeatureWindowDao.class, FeatureWindowDao_Impl.getRequiredConverters());
    _typeConvertersMap.put(StressLabelDao.class, StressLabelDao_Impl.getRequiredConverters());
    return _typeConvertersMap;
  }

  @Override
  @NonNull
  public Set<Class<? extends AutoMigrationSpec>> getRequiredAutoMigrationSpecs() {
    final HashSet<Class<? extends AutoMigrationSpec>> _autoMigrationSpecsSet = new HashSet<Class<? extends AutoMigrationSpec>>();
    return _autoMigrationSpecsSet;
  }

  @Override
  @NonNull
  public List<Migration> getAutoMigrations(
      @NonNull final Map<Class<? extends AutoMigrationSpec>, AutoMigrationSpec> autoMigrationSpecs) {
    final List<Migration> _autoMigrations = new ArrayList<Migration>();
    return _autoMigrations;
  }

  @Override
  public UserDao userDao() {
    if (_userDao != null) {
      return _userDao;
    } else {
      synchronized(this) {
        if(_userDao == null) {
          _userDao = new UserDao_Impl(this);
        }
        return _userDao;
      }
    }
  }

  @Override
  public SessionDao sessionDao() {
    if (_sessionDao != null) {
      return _sessionDao;
    } else {
      synchronized(this) {
        if(_sessionDao == null) {
          _sessionDao = new SessionDao_Impl(this);
        }
        return _sessionDao;
      }
    }
  }

  @Override
  public KeystrokeEventDao keystrokeEventDao() {
    if (_keystrokeEventDao != null) {
      return _keystrokeEventDao;
    } else {
      synchronized(this) {
        if(_keystrokeEventDao == null) {
          _keystrokeEventDao = new KeystrokeEventDao_Impl(this);
        }
        return _keystrokeEventDao;
      }
    }
  }

  @Override
  public FeatureWindowDao featureWindowDao() {
    if (_featureWindowDao != null) {
      return _featureWindowDao;
    } else {
      synchronized(this) {
        if(_featureWindowDao == null) {
          _featureWindowDao = new FeatureWindowDao_Impl(this);
        }
        return _featureWindowDao;
      }
    }
  }

  @Override
  public StressLabelDao stressLabelDao() {
    if (_stressLabelDao != null) {
      return _stressLabelDao;
    } else {
      synchronized(this) {
        if(_stressLabelDao == null) {
          _stressLabelDao = new StressLabelDao_Impl(this);
        }
        return _stressLabelDao;
      }
    }
  }
}
