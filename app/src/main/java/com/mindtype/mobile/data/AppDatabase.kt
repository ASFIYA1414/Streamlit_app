package com.mindtype.mobile.data

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import com.mindtype.mobile.data.dao.*
import com.mindtype.mobile.data.entity.*

@Database(
    entities = [
        UserEntity::class,
        SessionEntity::class,
        KeystrokeEventEntity::class,
        FeatureWindowEntity::class,
        StressLabelEntity::class
    ],
    version = 2,
    exportSchema = false
)
abstract class AppDatabase : RoomDatabase() {

    abstract fun userDao(): UserDao
    abstract fun sessionDao(): SessionDao
    abstract fun keystrokeEventDao(): KeystrokeEventDao
    abstract fun featureWindowDao(): FeatureWindowDao
    abstract fun stressLabelDao(): StressLabelDao

    companion object {
        @Volatile
        private var INSTANCE: AppDatabase? = null

        fun getInstance(context: Context): AppDatabase {
            return INSTANCE ?: synchronized(this) {
                val instance = Room.databaseBuilder(
                    context.applicationContext,
                    AppDatabase::class.java,
                    "mindtype_database"
                )
                    .fallbackToDestructiveMigration()
                    .build()
                INSTANCE = instance
                instance
            }
        }
    }
}
