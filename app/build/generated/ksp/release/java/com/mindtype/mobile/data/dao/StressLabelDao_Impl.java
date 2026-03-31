package com.mindtype.mobile.data.dao;

import android.database.Cursor;
import android.os.CancellationSignal;
import androidx.annotation.NonNull;
import androidx.room.CoroutinesRoom;
import androidx.room.EntityInsertionAdapter;
import androidx.room.RoomDatabase;
import androidx.room.RoomSQLiteQuery;
import androidx.room.util.CursorUtil;
import androidx.room.util.DBUtil;
import androidx.sqlite.db.SupportSQLiteStatement;
import com.mindtype.mobile.data.entity.StressLabelEntity;
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
public final class StressLabelDao_Impl implements StressLabelDao {
  private final RoomDatabase __db;

  private final EntityInsertionAdapter<StressLabelEntity> __insertionAdapterOfStressLabelEntity;

  public StressLabelDao_Impl(@NonNull final RoomDatabase __db) {
    this.__db = __db;
    this.__insertionAdapterOfStressLabelEntity = new EntityInsertionAdapter<StressLabelEntity>(__db) {
      @Override
      @NonNull
      protected String createQuery() {
        return "INSERT OR REPLACE INTO `stress_labels` (`label_id`,`session_id`,`timestamp`,`raw_score`,`mapped_class`) VALUES (nullif(?, 0),?,?,?,?)";
      }

      @Override
      protected void bind(@NonNull final SupportSQLiteStatement statement,
          @NonNull final StressLabelEntity entity) {
        statement.bindLong(1, entity.getLabelId());
        statement.bindString(2, entity.getSessionId());
        statement.bindLong(3, entity.getTimestamp());
        statement.bindLong(4, entity.getRawScore());
        statement.bindString(5, entity.getMappedClass());
      }
    };
  }

  @Override
  public Object insert(final StressLabelEntity label,
      final Continuation<? super Unit> $completion) {
    return CoroutinesRoom.execute(__db, true, new Callable<Unit>() {
      @Override
      @NonNull
      public Unit call() throws Exception {
        __db.beginTransaction();
        try {
          __insertionAdapterOfStressLabelEntity.insert(label);
          __db.setTransactionSuccessful();
          return Unit.INSTANCE;
        } finally {
          __db.endTransaction();
        }
      }
    }, $completion);
  }

  @Override
  public Object getLabelsForSession(final String sessionId,
      final Continuation<? super List<StressLabelEntity>> $completion) {
    final String _sql = "SELECT * FROM stress_labels WHERE session_id = ? ORDER BY timestamp ASC";
    final RoomSQLiteQuery _statement = RoomSQLiteQuery.acquire(_sql, 1);
    int _argIndex = 1;
    _statement.bindString(_argIndex, sessionId);
    final CancellationSignal _cancellationSignal = DBUtil.createCancellationSignal();
    return CoroutinesRoom.execute(__db, false, _cancellationSignal, new Callable<List<StressLabelEntity>>() {
      @Override
      @NonNull
      public List<StressLabelEntity> call() throws Exception {
        final Cursor _cursor = DBUtil.query(__db, _statement, false, null);
        try {
          final int _cursorIndexOfLabelId = CursorUtil.getColumnIndexOrThrow(_cursor, "label_id");
          final int _cursorIndexOfSessionId = CursorUtil.getColumnIndexOrThrow(_cursor, "session_id");
          final int _cursorIndexOfTimestamp = CursorUtil.getColumnIndexOrThrow(_cursor, "timestamp");
          final int _cursorIndexOfRawScore = CursorUtil.getColumnIndexOrThrow(_cursor, "raw_score");
          final int _cursorIndexOfMappedClass = CursorUtil.getColumnIndexOrThrow(_cursor, "mapped_class");
          final List<StressLabelEntity> _result = new ArrayList<StressLabelEntity>(_cursor.getCount());
          while (_cursor.moveToNext()) {
            final StressLabelEntity _item;
            final int _tmpLabelId;
            _tmpLabelId = _cursor.getInt(_cursorIndexOfLabelId);
            final String _tmpSessionId;
            _tmpSessionId = _cursor.getString(_cursorIndexOfSessionId);
            final long _tmpTimestamp;
            _tmpTimestamp = _cursor.getLong(_cursorIndexOfTimestamp);
            final int _tmpRawScore;
            _tmpRawScore = _cursor.getInt(_cursorIndexOfRawScore);
            final String _tmpMappedClass;
            _tmpMappedClass = _cursor.getString(_cursorIndexOfMappedClass);
            _item = new StressLabelEntity(_tmpLabelId,_tmpSessionId,_tmpTimestamp,_tmpRawScore,_tmpMappedClass);
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
  public Object getAllLabels(final Continuation<? super List<StressLabelEntity>> $completion) {
    final String _sql = "SELECT * FROM stress_labels ORDER BY timestamp ASC";
    final RoomSQLiteQuery _statement = RoomSQLiteQuery.acquire(_sql, 0);
    final CancellationSignal _cancellationSignal = DBUtil.createCancellationSignal();
    return CoroutinesRoom.execute(__db, false, _cancellationSignal, new Callable<List<StressLabelEntity>>() {
      @Override
      @NonNull
      public List<StressLabelEntity> call() throws Exception {
        final Cursor _cursor = DBUtil.query(__db, _statement, false, null);
        try {
          final int _cursorIndexOfLabelId = CursorUtil.getColumnIndexOrThrow(_cursor, "label_id");
          final int _cursorIndexOfSessionId = CursorUtil.getColumnIndexOrThrow(_cursor, "session_id");
          final int _cursorIndexOfTimestamp = CursorUtil.getColumnIndexOrThrow(_cursor, "timestamp");
          final int _cursorIndexOfRawScore = CursorUtil.getColumnIndexOrThrow(_cursor, "raw_score");
          final int _cursorIndexOfMappedClass = CursorUtil.getColumnIndexOrThrow(_cursor, "mapped_class");
          final List<StressLabelEntity> _result = new ArrayList<StressLabelEntity>(_cursor.getCount());
          while (_cursor.moveToNext()) {
            final StressLabelEntity _item;
            final int _tmpLabelId;
            _tmpLabelId = _cursor.getInt(_cursorIndexOfLabelId);
            final String _tmpSessionId;
            _tmpSessionId = _cursor.getString(_cursorIndexOfSessionId);
            final long _tmpTimestamp;
            _tmpTimestamp = _cursor.getLong(_cursorIndexOfTimestamp);
            final int _tmpRawScore;
            _tmpRawScore = _cursor.getInt(_cursorIndexOfRawScore);
            final String _tmpMappedClass;
            _tmpMappedClass = _cursor.getString(_cursorIndexOfMappedClass);
            _item = new StressLabelEntity(_tmpLabelId,_tmpSessionId,_tmpTimestamp,_tmpRawScore,_tmpMappedClass);
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
