package com.mindtype.mobile.data.dao;

import android.database.Cursor;
import android.os.CancellationSignal;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.room.CoroutinesRoom;
import androidx.room.EntityInsertionAdapter;
import androidx.room.RoomDatabase;
import androidx.room.RoomSQLiteQuery;
import androidx.room.SharedSQLiteStatement;
import androidx.room.util.CursorUtil;
import androidx.room.util.DBUtil;
import androidx.sqlite.db.SupportSQLiteStatement;
import com.mindtype.mobile.data.entity.FeatureWindowEntity;
import java.lang.Class;
import java.lang.Exception;
import java.lang.Object;
import java.lang.Override;
import java.lang.String;
import java.lang.SuppressWarnings;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.Callable;
import javax.annotation.processing.Generated;
import kotlin.Unit;
import kotlin.coroutines.Continuation;

@Generated("androidx.room.RoomProcessor")
@SuppressWarnings({"unchecked", "deprecation"})
public final class FeatureWindowDao_Impl implements FeatureWindowDao {
  private final RoomDatabase __db;

  private final EntityInsertionAdapter<FeatureWindowEntity> __insertionAdapterOfFeatureWindowEntity;

  private final SharedSQLiteStatement __preparedStmtOfUpdateStressLabel;

  public FeatureWindowDao_Impl(@NonNull final RoomDatabase __db) {
    this.__db = __db;
    this.__insertionAdapterOfFeatureWindowEntity = new EntityInsertionAdapter<FeatureWindowEntity>(__db) {
      @Override
      @NonNull
      protected String createQuery() {
        return "INSERT OR REPLACE INTO `feature_windows` (`window_id`,`session_id`,`window_start`,`window_end`,`mean_dwell`,`std_dwell`,`mean_flight`,`std_flight`,`typing_speed`,`backspace_rate`,`pause_count`,`mean_pressure`,`gyro_std`,`predicted_class`,`stress_label`) VALUES (nullif(?, 0),?,?,?,?,?,?,?,?,?,?,?,?,?,?)";
      }

      @Override
      protected void bind(@NonNull final SupportSQLiteStatement statement,
          @NonNull final FeatureWindowEntity entity) {
        statement.bindLong(1, entity.getWindowId());
        statement.bindString(2, entity.getSessionId());
        statement.bindLong(3, entity.getWindowStart());
        statement.bindLong(4, entity.getWindowEnd());
        statement.bindDouble(5, entity.getMeanDwell());
        statement.bindDouble(6, entity.getStdDwell());
        statement.bindDouble(7, entity.getMeanFlight());
        statement.bindDouble(8, entity.getStdFlight());
        statement.bindDouble(9, entity.getTypingSpeed());
        statement.bindDouble(10, entity.getBackspaceRate());
        statement.bindLong(11, entity.getPauseCount());
        statement.bindDouble(12, entity.getMeanPressure());
        statement.bindDouble(13, entity.getGyroStd());
        statement.bindString(14, entity.getPredictedClass());
        if (entity.getStressLabel() == null) {
          statement.bindNull(15);
        } else {
          statement.bindString(15, entity.getStressLabel());
        }
      }
    };
    this.__preparedStmtOfUpdateStressLabel = new SharedSQLiteStatement(__db) {
      @Override
      @NonNull
      public String createQuery() {
        final String _query = "UPDATE feature_windows SET stress_label = ? WHERE window_id = ?";
        return _query;
      }
    };
  }

  @Override
  public Object insert(final FeatureWindowEntity window,
      final Continuation<? super Unit> $completion) {
    return CoroutinesRoom.execute(__db, true, new Callable<Unit>() {
      @Override
      @NonNull
      public Unit call() throws Exception {
        __db.beginTransaction();
        try {
          __insertionAdapterOfFeatureWindowEntity.insert(window);
          __db.setTransactionSuccessful();
          return Unit.INSTANCE;
        } finally {
          __db.endTransaction();
        }
      }
    }, $completion);
  }

  @Override
  public Object updateStressLabel(final int windowId, final String label,
      final Continuation<? super Unit> $completion) {
    return CoroutinesRoom.execute(__db, true, new Callable<Unit>() {
      @Override
      @NonNull
      public Unit call() throws Exception {
        final SupportSQLiteStatement _stmt = __preparedStmtOfUpdateStressLabel.acquire();
        int _argIndex = 1;
        _stmt.bindString(_argIndex, label);
        _argIndex = 2;
        _stmt.bindLong(_argIndex, windowId);
        try {
          __db.beginTransaction();
          try {
            _stmt.executeUpdateDelete();
            __db.setTransactionSuccessful();
            return Unit.INSTANCE;
          } finally {
            __db.endTransaction();
          }
        } finally {
          __preparedStmtOfUpdateStressLabel.release(_stmt);
        }
      }
    }, $completion);
  }

  @Override
  public Object getWindowsForSession(final String sessionId,
      final Continuation<? super List<FeatureWindowEntity>> $completion) {
    final String _sql = "SELECT * FROM feature_windows WHERE session_id = ? ORDER BY window_start ASC";
    final RoomSQLiteQuery _statement = RoomSQLiteQuery.acquire(_sql, 1);
    int _argIndex = 1;
    _statement.bindString(_argIndex, sessionId);
    final CancellationSignal _cancellationSignal = DBUtil.createCancellationSignal();
    return CoroutinesRoom.execute(__db, false, _cancellationSignal, new Callable<List<FeatureWindowEntity>>() {
      @Override
      @NonNull
      public List<FeatureWindowEntity> call() throws Exception {
        final Cursor _cursor = DBUtil.query(__db, _statement, false, null);
        try {
          final int _cursorIndexOfWindowId = CursorUtil.getColumnIndexOrThrow(_cursor, "window_id");
          final int _cursorIndexOfSessionId = CursorUtil.getColumnIndexOrThrow(_cursor, "session_id");
          final int _cursorIndexOfWindowStart = CursorUtil.getColumnIndexOrThrow(_cursor, "window_start");
          final int _cursorIndexOfWindowEnd = CursorUtil.getColumnIndexOrThrow(_cursor, "window_end");
          final int _cursorIndexOfMeanDwell = CursorUtil.getColumnIndexOrThrow(_cursor, "mean_dwell");
          final int _cursorIndexOfStdDwell = CursorUtil.getColumnIndexOrThrow(_cursor, "std_dwell");
          final int _cursorIndexOfMeanFlight = CursorUtil.getColumnIndexOrThrow(_cursor, "mean_flight");
          final int _cursorIndexOfStdFlight = CursorUtil.getColumnIndexOrThrow(_cursor, "std_flight");
          final int _cursorIndexOfTypingSpeed = CursorUtil.getColumnIndexOrThrow(_cursor, "typing_speed");
          final int _cursorIndexOfBackspaceRate = CursorUtil.getColumnIndexOrThrow(_cursor, "backspace_rate");
          final int _cursorIndexOfPauseCount = CursorUtil.getColumnIndexOrThrow(_cursor, "pause_count");
          final int _cursorIndexOfMeanPressure = CursorUtil.getColumnIndexOrThrow(_cursor, "mean_pressure");
          final int _cursorIndexOfGyroStd = CursorUtil.getColumnIndexOrThrow(_cursor, "gyro_std");
          final int _cursorIndexOfPredictedClass = CursorUtil.getColumnIndexOrThrow(_cursor, "predicted_class");
          final int _cursorIndexOfStressLabel = CursorUtil.getColumnIndexOrThrow(_cursor, "stress_label");
          final List<FeatureWindowEntity> _result = new ArrayList<FeatureWindowEntity>(_cursor.getCount());
          while (_cursor.moveToNext()) {
            final FeatureWindowEntity _item;
            final int _tmpWindowId;
            _tmpWindowId = _cursor.getInt(_cursorIndexOfWindowId);
            final String _tmpSessionId;
            _tmpSessionId = _cursor.getString(_cursorIndexOfSessionId);
            final long _tmpWindowStart;
            _tmpWindowStart = _cursor.getLong(_cursorIndexOfWindowStart);
            final long _tmpWindowEnd;
            _tmpWindowEnd = _cursor.getLong(_cursorIndexOfWindowEnd);
            final float _tmpMeanDwell;
            _tmpMeanDwell = _cursor.getFloat(_cursorIndexOfMeanDwell);
            final float _tmpStdDwell;
            _tmpStdDwell = _cursor.getFloat(_cursorIndexOfStdDwell);
            final float _tmpMeanFlight;
            _tmpMeanFlight = _cursor.getFloat(_cursorIndexOfMeanFlight);
            final float _tmpStdFlight;
            _tmpStdFlight = _cursor.getFloat(_cursorIndexOfStdFlight);
            final float _tmpTypingSpeed;
            _tmpTypingSpeed = _cursor.getFloat(_cursorIndexOfTypingSpeed);
            final float _tmpBackspaceRate;
            _tmpBackspaceRate = _cursor.getFloat(_cursorIndexOfBackspaceRate);
            final int _tmpPauseCount;
            _tmpPauseCount = _cursor.getInt(_cursorIndexOfPauseCount);
            final float _tmpMeanPressure;
            _tmpMeanPressure = _cursor.getFloat(_cursorIndexOfMeanPressure);
            final float _tmpGyroStd;
            _tmpGyroStd = _cursor.getFloat(_cursorIndexOfGyroStd);
            final String _tmpPredictedClass;
            _tmpPredictedClass = _cursor.getString(_cursorIndexOfPredictedClass);
            final String _tmpStressLabel;
            if (_cursor.isNull(_cursorIndexOfStressLabel)) {
              _tmpStressLabel = null;
            } else {
              _tmpStressLabel = _cursor.getString(_cursorIndexOfStressLabel);
            }
            _item = new FeatureWindowEntity(_tmpWindowId,_tmpSessionId,_tmpWindowStart,_tmpWindowEnd,_tmpMeanDwell,_tmpStdDwell,_tmpMeanFlight,_tmpStdFlight,_tmpTypingSpeed,_tmpBackspaceRate,_tmpPauseCount,_tmpMeanPressure,_tmpGyroStd,_tmpPredictedClass,_tmpStressLabel);
            _result.add(_item);
          }
          return _result;
        } finally {
          _cursor.close();
          _statement.release();
        }
      }
    }, $completion);
  }

  @Override
  public Object getWindowsSince(final long sinceMs,
      final Continuation<? super List<FeatureWindowEntity>> $completion) {
    final String _sql = "SELECT * FROM feature_windows WHERE window_start >= ? ORDER BY window_start ASC";
    final RoomSQLiteQuery _statement = RoomSQLiteQuery.acquire(_sql, 1);
    int _argIndex = 1;
    _statement.bindLong(_argIndex, sinceMs);
    final CancellationSignal _cancellationSignal = DBUtil.createCancellationSignal();
    return CoroutinesRoom.execute(__db, false, _cancellationSignal, new Callable<List<FeatureWindowEntity>>() {
      @Override
      @NonNull
      public List<FeatureWindowEntity> call() throws Exception {
        final Cursor _cursor = DBUtil.query(__db, _statement, false, null);
        try {
          final int _cursorIndexOfWindowId = CursorUtil.getColumnIndexOrThrow(_cursor, "window_id");
          final int _cursorIndexOfSessionId = CursorUtil.getColumnIndexOrThrow(_cursor, "session_id");
          final int _cursorIndexOfWindowStart = CursorUtil.getColumnIndexOrThrow(_cursor, "window_start");
          final int _cursorIndexOfWindowEnd = CursorUtil.getColumnIndexOrThrow(_cursor, "window_end");
          final int _cursorIndexOfMeanDwell = CursorUtil.getColumnIndexOrThrow(_cursor, "mean_dwell");
          final int _cursorIndexOfStdDwell = CursorUtil.getColumnIndexOrThrow(_cursor, "std_dwell");
          final int _cursorIndexOfMeanFlight = CursorUtil.getColumnIndexOrThrow(_cursor, "mean_flight");
          final int _cursorIndexOfStdFlight = CursorUtil.getColumnIndexOrThrow(_cursor, "std_flight");
          final int _cursorIndexOfTypingSpeed = CursorUtil.getColumnIndexOrThrow(_cursor, "typing_speed");
          final int _cursorIndexOfBackspaceRate = CursorUtil.getColumnIndexOrThrow(_cursor, "backspace_rate");
          final int _cursorIndexOfPauseCount = CursorUtil.getColumnIndexOrThrow(_cursor, "pause_count");
          final int _cursorIndexOfMeanPressure = CursorUtil.getColumnIndexOrThrow(_cursor, "mean_pressure");
          final int _cursorIndexOfGyroStd = CursorUtil.getColumnIndexOrThrow(_cursor, "gyro_std");
          final int _cursorIndexOfPredictedClass = CursorUtil.getColumnIndexOrThrow(_cursor, "predicted_class");
          final int _cursorIndexOfStressLabel = CursorUtil.getColumnIndexOrThrow(_cursor, "stress_label");
          final List<FeatureWindowEntity> _result = new ArrayList<FeatureWindowEntity>(_cursor.getCount());
          while (_cursor.moveToNext()) {
            final FeatureWindowEntity _item;
            final int _tmpWindowId;
            _tmpWindowId = _cursor.getInt(_cursorIndexOfWindowId);
            final String _tmpSessionId;
            _tmpSessionId = _cursor.getString(_cursorIndexOfSessionId);
            final long _tmpWindowStart;
            _tmpWindowStart = _cursor.getLong(_cursorIndexOfWindowStart);
            final long _tmpWindowEnd;
            _tmpWindowEnd = _cursor.getLong(_cursorIndexOfWindowEnd);
            final float _tmpMeanDwell;
            _tmpMeanDwell = _cursor.getFloat(_cursorIndexOfMeanDwell);
            final float _tmpStdDwell;
            _tmpStdDwell = _cursor.getFloat(_cursorIndexOfStdDwell);
            final float _tmpMeanFlight;
            _tmpMeanFlight = _cursor.getFloat(_cursorIndexOfMeanFlight);
            final float _tmpStdFlight;
            _tmpStdFlight = _cursor.getFloat(_cursorIndexOfStdFlight);
            final float _tmpTypingSpeed;
            _tmpTypingSpeed = _cursor.getFloat(_cursorIndexOfTypingSpeed);
            final float _tmpBackspaceRate;
            _tmpBackspaceRate = _cursor.getFloat(_cursorIndexOfBackspaceRate);
            final int _tmpPauseCount;
            _tmpPauseCount = _cursor.getInt(_cursorIndexOfPauseCount);
            final float _tmpMeanPressure;
            _tmpMeanPressure = _cursor.getFloat(_cursorIndexOfMeanPressure);
            final float _tmpGyroStd;
            _tmpGyroStd = _cursor.getFloat(_cursorIndexOfGyroStd);
            final String _tmpPredictedClass;
            _tmpPredictedClass = _cursor.getString(_cursorIndexOfPredictedClass);
            final String _tmpStressLabel;
            if (_cursor.isNull(_cursorIndexOfStressLabel)) {
              _tmpStressLabel = null;
            } else {
              _tmpStressLabel = _cursor.getString(_cursorIndexOfStressLabel);
            }
            _item = new FeatureWindowEntity(_tmpWindowId,_tmpSessionId,_tmpWindowStart,_tmpWindowEnd,_tmpMeanDwell,_tmpStdDwell,_tmpMeanFlight,_tmpStdFlight,_tmpTypingSpeed,_tmpBackspaceRate,_tmpPauseCount,_tmpMeanPressure,_tmpGyroStd,_tmpPredictedClass,_tmpStressLabel);
            _result.add(_item);
          }
          return _result;
        } finally {
          _cursor.close();
          _statement.release();
        }
      }
    }, $completion);
  }

  @Override
  public Object getLatestWindow(final Continuation<? super FeatureWindowEntity> $completion) {
    final String _sql = "SELECT * FROM feature_windows ORDER BY window_start DESC LIMIT 1";
    final RoomSQLiteQuery _statement = RoomSQLiteQuery.acquire(_sql, 0);
    final CancellationSignal _cancellationSignal = DBUtil.createCancellationSignal();
    return CoroutinesRoom.execute(__db, false, _cancellationSignal, new Callable<FeatureWindowEntity>() {
      @Override
      @Nullable
      public FeatureWindowEntity call() throws Exception {
        final Cursor _cursor = DBUtil.query(__db, _statement, false, null);
        try {
          final int _cursorIndexOfWindowId = CursorUtil.getColumnIndexOrThrow(_cursor, "window_id");
          final int _cursorIndexOfSessionId = CursorUtil.getColumnIndexOrThrow(_cursor, "session_id");
          final int _cursorIndexOfWindowStart = CursorUtil.getColumnIndexOrThrow(_cursor, "window_start");
          final int _cursorIndexOfWindowEnd = CursorUtil.getColumnIndexOrThrow(_cursor, "window_end");
          final int _cursorIndexOfMeanDwell = CursorUtil.getColumnIndexOrThrow(_cursor, "mean_dwell");
          final int _cursorIndexOfStdDwell = CursorUtil.getColumnIndexOrThrow(_cursor, "std_dwell");
          final int _cursorIndexOfMeanFlight = CursorUtil.getColumnIndexOrThrow(_cursor, "mean_flight");
          final int _cursorIndexOfStdFlight = CursorUtil.getColumnIndexOrThrow(_cursor, "std_flight");
          final int _cursorIndexOfTypingSpeed = CursorUtil.getColumnIndexOrThrow(_cursor, "typing_speed");
          final int _cursorIndexOfBackspaceRate = CursorUtil.getColumnIndexOrThrow(_cursor, "backspace_rate");
          final int _cursorIndexOfPauseCount = CursorUtil.getColumnIndexOrThrow(_cursor, "pause_count");
          final int _cursorIndexOfMeanPressure = CursorUtil.getColumnIndexOrThrow(_cursor, "mean_pressure");
          final int _cursorIndexOfGyroStd = CursorUtil.getColumnIndexOrThrow(_cursor, "gyro_std");
          final int _cursorIndexOfPredictedClass = CursorUtil.getColumnIndexOrThrow(_cursor, "predicted_class");
          final int _cursorIndexOfStressLabel = CursorUtil.getColumnIndexOrThrow(_cursor, "stress_label");
          final FeatureWindowEntity _result;
          if (_cursor.moveToFirst()) {
            final int _tmpWindowId;
            _tmpWindowId = _cursor.getInt(_cursorIndexOfWindowId);
            final String _tmpSessionId;
            _tmpSessionId = _cursor.getString(_cursorIndexOfSessionId);
            final long _tmpWindowStart;
            _tmpWindowStart = _cursor.getLong(_cursorIndexOfWindowStart);
            final long _tmpWindowEnd;
            _tmpWindowEnd = _cursor.getLong(_cursorIndexOfWindowEnd);
            final float _tmpMeanDwell;
            _tmpMeanDwell = _cursor.getFloat(_cursorIndexOfMeanDwell);
            final float _tmpStdDwell;
            _tmpStdDwell = _cursor.getFloat(_cursorIndexOfStdDwell);
            final float _tmpMeanFlight;
            _tmpMeanFlight = _cursor.getFloat(_cursorIndexOfMeanFlight);
            final float _tmpStdFlight;
            _tmpStdFlight = _cursor.getFloat(_cursorIndexOfStdFlight);
            final float _tmpTypingSpeed;
            _tmpTypingSpeed = _cursor.getFloat(_cursorIndexOfTypingSpeed);
            final float _tmpBackspaceRate;
            _tmpBackspaceRate = _cursor.getFloat(_cursorIndexOfBackspaceRate);
            final int _tmpPauseCount;
            _tmpPauseCount = _cursor.getInt(_cursorIndexOfPauseCount);
            final float _tmpMeanPressure;
            _tmpMeanPressure = _cursor.getFloat(_cursorIndexOfMeanPressure);
            final float _tmpGyroStd;
            _tmpGyroStd = _cursor.getFloat(_cursorIndexOfGyroStd);
            final String _tmpPredictedClass;
            _tmpPredictedClass = _cursor.getString(_cursorIndexOfPredictedClass);
            final String _tmpStressLabel;
            if (_cursor.isNull(_cursorIndexOfStressLabel)) {
              _tmpStressLabel = null;
            } else {
              _tmpStressLabel = _cursor.getString(_cursorIndexOfStressLabel);
            }
            _result = new FeatureWindowEntity(_tmpWindowId,_tmpSessionId,_tmpWindowStart,_tmpWindowEnd,_tmpMeanDwell,_tmpStdDwell,_tmpMeanFlight,_tmpStdFlight,_tmpTypingSpeed,_tmpBackspaceRate,_tmpPauseCount,_tmpMeanPressure,_tmpGyroStd,_tmpPredictedClass,_tmpStressLabel);
          } else {
            _result = null;
          }
          return _result;
        } finally {
          _cursor.close();
          _statement.release();
        }
      }
    }, $completion);
  }

  @Override
  public Object getAllWindows(final Continuation<? super List<FeatureWindowEntity>> $completion) {
    final String _sql = "SELECT * FROM feature_windows ORDER BY window_start ASC";
    final RoomSQLiteQuery _statement = RoomSQLiteQuery.acquire(_sql, 0);
    final CancellationSignal _cancellationSignal = DBUtil.createCancellationSignal();
    return CoroutinesRoom.execute(__db, false, _cancellationSignal, new Callable<List<FeatureWindowEntity>>() {
      @Override
      @NonNull
      public List<FeatureWindowEntity> call() throws Exception {
        final Cursor _cursor = DBUtil.query(__db, _statement, false, null);
        try {
          final int _cursorIndexOfWindowId = CursorUtil.getColumnIndexOrThrow(_cursor, "window_id");
          final int _cursorIndexOfSessionId = CursorUtil.getColumnIndexOrThrow(_cursor, "session_id");
          final int _cursorIndexOfWindowStart = CursorUtil.getColumnIndexOrThrow(_cursor, "window_start");
          final int _cursorIndexOfWindowEnd = CursorUtil.getColumnIndexOrThrow(_cursor, "window_end");
          final int _cursorIndexOfMeanDwell = CursorUtil.getColumnIndexOrThrow(_cursor, "mean_dwell");
          final int _cursorIndexOfStdDwell = CursorUtil.getColumnIndexOrThrow(_cursor, "std_dwell");
          final int _cursorIndexOfMeanFlight = CursorUtil.getColumnIndexOrThrow(_cursor, "mean_flight");
          final int _cursorIndexOfStdFlight = CursorUtil.getColumnIndexOrThrow(_cursor, "std_flight");
          final int _cursorIndexOfTypingSpeed = CursorUtil.getColumnIndexOrThrow(_cursor, "typing_speed");
          final int _cursorIndexOfBackspaceRate = CursorUtil.getColumnIndexOrThrow(_cursor, "backspace_rate");
          final int _cursorIndexOfPauseCount = CursorUtil.getColumnIndexOrThrow(_cursor, "pause_count");
          final int _cursorIndexOfMeanPressure = CursorUtil.getColumnIndexOrThrow(_cursor, "mean_pressure");
          final int _cursorIndexOfGyroStd = CursorUtil.getColumnIndexOrThrow(_cursor, "gyro_std");
          final int _cursorIndexOfPredictedClass = CursorUtil.getColumnIndexOrThrow(_cursor, "predicted_class");
          final int _cursorIndexOfStressLabel = CursorUtil.getColumnIndexOrThrow(_cursor, "stress_label");
          final List<FeatureWindowEntity> _result = new ArrayList<FeatureWindowEntity>(_cursor.getCount());
          while (_cursor.moveToNext()) {
            final FeatureWindowEntity _item;
            final int _tmpWindowId;
            _tmpWindowId = _cursor.getInt(_cursorIndexOfWindowId);
            final String _tmpSessionId;
            _tmpSessionId = _cursor.getString(_cursorIndexOfSessionId);
            final long _tmpWindowStart;
            _tmpWindowStart = _cursor.getLong(_cursorIndexOfWindowStart);
            final long _tmpWindowEnd;
            _tmpWindowEnd = _cursor.getLong(_cursorIndexOfWindowEnd);
            final float _tmpMeanDwell;
            _tmpMeanDwell = _cursor.getFloat(_cursorIndexOfMeanDwell);
            final float _tmpStdDwell;
            _tmpStdDwell = _cursor.getFloat(_cursorIndexOfStdDwell);
            final float _tmpMeanFlight;
            _tmpMeanFlight = _cursor.getFloat(_cursorIndexOfMeanFlight);
            final float _tmpStdFlight;
            _tmpStdFlight = _cursor.getFloat(_cursorIndexOfStdFlight);
            final float _tmpTypingSpeed;
            _tmpTypingSpeed = _cursor.getFloat(_cursorIndexOfTypingSpeed);
            final float _tmpBackspaceRate;
            _tmpBackspaceRate = _cursor.getFloat(_cursorIndexOfBackspaceRate);
            final int _tmpPauseCount;
            _tmpPauseCount = _cursor.getInt(_cursorIndexOfPauseCount);
            final float _tmpMeanPressure;
            _tmpMeanPressure = _cursor.getFloat(_cursorIndexOfMeanPressure);
            final float _tmpGyroStd;
            _tmpGyroStd = _cursor.getFloat(_cursorIndexOfGyroStd);
            final String _tmpPredictedClass;
            _tmpPredictedClass = _cursor.getString(_cursorIndexOfPredictedClass);
            final String _tmpStressLabel;
            if (_cursor.isNull(_cursorIndexOfStressLabel)) {
              _tmpStressLabel = null;
            } else {
              _tmpStressLabel = _cursor.getString(_cursorIndexOfStressLabel);
            }
            _item = new FeatureWindowEntity(_tmpWindowId,_tmpSessionId,_tmpWindowStart,_tmpWindowEnd,_tmpMeanDwell,_tmpStdDwell,_tmpMeanFlight,_tmpStdFlight,_tmpTypingSpeed,_tmpBackspaceRate,_tmpPauseCount,_tmpMeanPressure,_tmpGyroStd,_tmpPredictedClass,_tmpStressLabel);
            _result.add(_item);
          }
          return _result;
        } finally {
          _cursor.close();
          _statement.release();
        }
      }
    }, $completion);
  }

  @NonNull
  public static List<Class<?>> getRequiredConverters() {
    return Collections.emptyList();
  }
}
