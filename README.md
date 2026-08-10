Making a self checkout terminal that uses facial recognition in order to detect the user and just by the user in frame, match to database and log in automatically using their own credits
This is mainly made for small stands or shops that don't have good internet access. All price predictions and face analysing is done locally.

Also includes a barcode scanner, user can scan any item and it would automaticaly search a international barcode database in order to retrieve the exact item.

However for price to reduce the use of the internet, by using a price algorithm or AI that would predict how much a item should cost in theory. So if it was a bag of doritos it would think a fair price to pay is $3.5 and maybe when stock is low increase the price. If AI controlled the price, it would be way smarter in balancing user experience vs actual profits in a real situation.

So user can theoretically just show face until it verifies its real and then scan item and then just walk off if transaction completed successfully. Removing the absolute friction of a low value self checkout process (like foods). High value purcahses should require more verification in practice.

Problems are mainly security, so i added a liveness detector that if the face is moving in a realistic pattern it would mark it as verified, this is not as accurate so the indicator is just used primarily for high value transactions only (theoretically)

I like this idea as i want to experiment if truely just cameras on a phone/ or laptop are enough to replace dedicated self checkouts all together, and make things more seamless.

## How it works.

IT takes a frame from the webcam and converts it from OpenCV's BGR colour order to RGB, which face_recognition and MediaPipe require.

For identification, face_recognition detects the face and produces a 128-number encoding of it. That encoding is compared against every stored encoding, and the closest match within the tolerance is returned as the user.

For liveness, MediaPipe Face Mesh returnslandmark points on the face. From fixed landmark indices I calculate two ratios: the mouth aspect ratio (mouth height ÷ width) and the eye aspect ratio (eye height ÷ width). These are collected certain window of webcam frames.

The check uses the variance of the MAR rather than its average, because resting mouth shape differs between people, variance measures whether the value is changing, not what it is. A live face constantly moves, so its variance is above the threshold. 

A photo held to the camera gives an identical value every frame, so its variance is near zero and it fails. A blink is also "required", detected when the EAR drops below a threshold. Only if both pass is the face marked live, and that result isn't re-tested during the same transaction. Then transaction is allowed to proceed
Usually only works if mouth is open.


## Improvements

Using mouth to verify transaction is kinda weird so calculate both EAR first and MAR as backup or altogether to make it more seamless
## Modules used

I have mainly used PyQt5 for the GUI, as its faster than other alternatives like tkinter, however does have disadvantages of its very inefficient way of creating elements (takes atlesat 5 lines to create one button), so maybe create a class that makes it more simple to add UI elements to my interface.

Pyzbar is a library primarly used to find barcodes and scan them.
