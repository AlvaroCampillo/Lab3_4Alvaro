package com.alvaro.lab3_4alvaro

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Camera
import android.net.Uri
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.provider.MediaStore
import android.provider.MediaStore.ACTION_IMAGE_CAPTURE
import android.provider.MediaStore.Images.Media.getBitmap
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.app.ActivityCompat
import com.alvaro.lab3_4alvaro.ml.MobilenetV110224Quant
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
// I used an external database for this project since after attempting multiple times
// ,following the exact same steps of the pdf provided,
// to create and run Firebase ML kit it gave an error that I couldnt solve.
class MainActivity : AppCompatActivity() {
    val requestcamera_code=100
    lateinit var select:Button
    lateinit var bitmap: Bitmap
    lateinit var imageview: ImageView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        imageview=findViewById(R.id.imageView)
        val fileName="labels.txt"
        val inputString=application.assets.open(fileName).bufferedReader().use { it.readText() }
        val townList =inputString.split("\n")

        var tv:TextView=findViewById(R.id.textView)
        var tv1:TextView=findViewById(R.id.textView1)
        var tv2: TextView =findViewById(R.id.textView3)

        var select: Button=findViewById(R.id.button)

        //The commented lines within this scope are used to switch between gallery and camera
        // As my computer cannot run the camera properly I only write the code

        select.setOnClickListener(View.OnClickListener {
            // -> GALLERY
            var intent:Intent = Intent(Intent.ACTION_GET_CONTENT)
            intent.type="image/*"
            startActivityForResult(intent,100)
            // -> CAMERA
            //var intent:Intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            //startActivityForResult(intent,requestcamera_code)

        })

        var predict:Button=findViewById(R.id.button2)
        predict.setOnClickListener(View.OnClickListener {
                                                        //bitmap
            var resized:Bitmap=Bitmap.createScaledBitmap(bitmap,224 ,224,true)

            val model = MobilenetV110224Quant.newInstance(this)

            // Creates inputs for reference.
            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.UINT8)
            //create byte buffer image
            var tbuffer = TensorImage.fromBitmap(resized)
            var byteBuffer = tbuffer.buffer

            inputFeature0.loadBuffer(byteBuffer)

            // Runs model inference and gets result.
            val outputs = model.process(inputFeature0)
            val outputFeature0 = outputs.outputFeature0AsTensorBuffer

            var max=getMax(outputFeature0.floatArray)

            // on this part we establish an order from most probable coincidence to less probable
            tv.setText(townList[max])
            tv1.setText(townList[max-1])
            tv2.setText(townList[max-2])

            // Releases model resources if no longer used.
            model.close()
        })


    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        // -> GALLERY
        imageview.setImageURI(data?.data)
        // store image to bitmap
        var uri:Uri?=data?.data
        bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver,uri)

        // Like I commented before to switch the input to the camera we have to change the
        // commented lines below with the ones above

        // -> CAMERA
        /*if (requestCode == requestcamera_code) {
            val imageBitmap = data?.extras?.get("data") as Bitmap
            imageview.setImageBitmap(imageBitmap)
            }*/

    }
    fun getMax(arr:FloatArray):Int{
        var ind=0
        var min=0.0f
        for(i in 0..1000)
        {
            if(arr[i]>min){
                ind=i
                min=arr[i]
            }
        }
        return ind
    }
}

