package com.mindtype.mobile.ui

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.mindtype.mobile.databinding.ActivitySettingsBinding
import com.mindtype.mobile.export.DataExporter
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class SettingsActivity : AppCompatActivity() {

    private lateinit var binding: ActivitySettingsBinding
    private var pendingCsvContent: String? = null

    // Android's built-in Save File dialog — zero permissions needed, never crashes
    private val createFileLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == Activity.RESULT_OK) {
            result.data?.data?.let { uri ->
                writeCsvToUri(uri)
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivitySettingsBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val prefs = getSharedPreferences("mindtype_prefs", Context.MODE_PRIVATE)
        val userId = prefs.getString("user_id", "—") ?: "—"
        binding.tvCurrentUserId.text = "Participant ID: $userId"

        binding.btnExportData.setOnClickListener {
            lifecycleScope.launch {
                binding.btnExportData.isEnabled = false
                binding.btnExportData.text = "Preparing…"

                // Build CSV string in background thread
                val csv = withContext(Dispatchers.IO) {
                    DataExporter(applicationContext).buildCsvString()
                }

                binding.btnExportData.isEnabled = true
                binding.btnExportData.text = "Export Data"

                if (csv == null) {
                    Toast.makeText(
                        this@SettingsActivity,
                        "No data yet — type for a few minutes with MindType keyboard first!",
                        Toast.LENGTH_LONG
                    ).show()
                    return@launch
                }

                // Store the CSV string and open system Save dialog
                pendingCsvContent = csv
                val intent = Intent(Intent.ACTION_CREATE_DOCUMENT).apply {
                    addCategory(Intent.CATEGORY_OPENABLE)
                    type = "text/csv"
                    putExtra(Intent.EXTRA_TITLE, "mindtype_dataset_${userId}.csv")
                }
                createFileLauncher.launch(intent)
            }
        }

        binding.tvPrivacyNote.text = "Privacy Reminder:\n• No typed text is stored\n• Data never leaves your device\n• Uninstall the app to delete all data"
    }

    private fun writeCsvToUri(uri: Uri) {
        val csv = pendingCsvContent ?: return
        try {
            contentResolver.openOutputStream(uri)?.use { stream ->
                stream.write(csv.toByteArray())
            }
            Toast.makeText(this, "✅ Dataset saved! Share the file with your researcher.", Toast.LENGTH_LONG).show()
            pendingCsvContent = null
        } catch (e: Exception) {
            Toast.makeText(this, "Save failed: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }
}
