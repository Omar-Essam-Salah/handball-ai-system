أنت خبير في بناء أنظمة Computer Vision و AI للرياضة. 

أنا عايز أبني نظام تحليل مباريات **كرة اليد** كامل Offline يعمل على لابتوب Dell Precision 7740 (i9 + RTX 3000 + 16GB+ RAM).

### متطلبات المشروع بالتفصيل:

**الكاميرا:**
- Hikvision DS-2CD2387G2-LU (8MP ColorVu, عدسة 2.8mm)
- متصلة باللابتوب عبر RTSP
- صورة حادة، low motion blur، ألوان واضحة (مهم جدًا لتمييز القمصان الأحمر والأزرق)

**النظام يجب أن يعمل:**
- Offline 100% أثناء المباراة (عبر Hotspot محلي)
- Live analysis أثناء المباراة
- إشعارات فورية + ملخص كل 5 دقائق على الموبايل + ساعة ذكية

**الوظائف المطلوبة:**
1. اكتشاف اللاعبين + تتبعهم (Player Detection + Tracking)
2. اكتشاف الكرة (Ball Detection + Tracking) — مهم جدًا لأنها صغيرة وسريعة
3. تمييز الفريقين باللون (Red Team = فريقي، Blue Team = المنافس)
4. تقسيم الملعب إلى مناطق (Left Wing, Left Back, Center, Right Back, Right Wing, Pivot Area 6m, 9m line, إلخ)
5. اكتشاف الأحداث: تسديدة، هدف، بداية هجمة، فقدان كرة، بلوك، 7 متر...
6. تحليل خاص لحارس المرمى: زاوية التسديد، مكان دخول الكرة في المرمى، نسبة التصدي
7. Heat Maps لمناطق الخطورة
8. إشعارات ذكية (مثال: "المنافس 65% هجوم من الباك الشمال" أو "3 تسديدات متتالية من الجناح")

**التقنيات المطلوب استخدامها:**
- Python
- OpenCV + Ultralytics YOLOv8 (يفضل YOLOv8m أو s مع RTX 3000)
- DeepSORT أو BoT-SORT للتتبع
- Homography أو Perspective Transform لتقسيم الملعب
- Color Detection (HSV) لتمييز الفرق
- Flask أو FastAPI + WebSocket للـ Dashboard والإشعارات المحلية
- Optional: Jersey Number Recognition (OCR) + Manual Mapping في البداية

**الجهاز:**
- Dell Precision 7740 مع NVIDIA RTX 3000 (CUDA متاح)
- هنستخدم 1080p @ 25-30 fps للحفاظ على السرعة

**المراحل المطلوبة:**
1. مرحلة 1: RTSP Stream + YOLO Detection + Tracking + Team Color
2. مرحلة 2: Court Mapping + Event Detection + Shots Analysis
3. مرحلة 3: Goalkeeper Specific Analysis + Notifications + Heatmaps
4. مرحلة 4: Full Tactical Analysis + Reports

الآن أريد منك:

أولاً: اكتب **Architecture كاملة** واضحة ومنظمة للمشروع (Pipeline، المكونات، كيفية الاتصال بينها).

ثانياً: خطة تطوير خطوة بخطوة (ما نبدأ بيه أولاً، والكود الأساسي لكل مرحلة).

ثالثاً: أفضل طريقة لقراءة الـ RTSP من الكاميرا + استخدام CUDA على RTX 3000.

رابعاً: نصائح مهمة لتقليل motion blur وتحسين دقة اكتشاف الكرة.

أخيرًا: لخص **كل** الرد السابق بأسلوب **Caveman** (بسيط جدًا، قصير، مثل رجل الكهف يتكلم: كلمات قليلة، مباشرة، بدون تعقيد).

ابدأ الآن.أنت خبير في بناء أنظمة Computer Vision و AI للرياضة. 

أنا عايز أبني نظام تحليل مباريات **كرة اليد** كامل Offline يعمل على لابتوب Dell Precision 7740 (i9 + RTX 3000 + 16GB+ RAM).

### متطلبات المشروع بالتفصيل:

**الكاميرا:**
- Hikvision DS-2CD2387G2-LU (8MP ColorVu, عدسة 2.8mm)
- متصلة باللابتوب عبر RTSP
- صورة حادة، low motion blur، ألوان واضحة (مهم جدًا لتمييز القمصان الأحمر والأزرق)

**النظام يجب أن يعمل:**
- Offline 100% أثناء المباراة (عبر Hotspot محلي)
- Live analysis أثناء المباراة
- إشعارات فورية + ملخص كل 5 دقائق على الموبايل + ساعة ذكية

**الوظائف المطلوبة:**
1. اكتشاف اللاعبين + تتبعهم (Player Detection + Tracking)
2. اكتشاف الكرة (Ball Detection + Tracking) — مهم جدًا لأنها صغيرة وسريعة
3. تمييز الفريقين باللون (Red Team = فريقي، Blue Team = المنافس)
4. تقسيم الملعب إلى مناطق (Left Wing, Left Back, Center, Right Back, Right Wing, Pivot Area 6m, 9m line, إلخ)
5. اكتشاف الأحداث: تسديدة، هدف، بداية هجمة، فقدان كرة، بلوك، 7 متر...
6. تحليل خاص لحارس المرمى: زاوية التسديد، مكان دخول الكرة في المرمى، نسبة التصدي
7. Heat Maps لمناطق الخطورة
8. إشعارات ذكية (مثال: "المنافس 65% هجوم من الباك الشمال" أو "3 تسديدات متتالية من الجناح")

**التقنيات المطلوب استخدامها:**
- Python
- OpenCV + Ultralytics YOLOv8 (يفضل YOLOv8m أو s مع RTX 3000)
- DeepSORT أو BoT-SORT للتتبع
- Homography أو Perspective Transform لتقسيم الملعب
- Color Detection (HSV) لتمييز الفرق
- Flask أو FastAPI + WebSocket للـ Dashboard والإشعارات المحلية
- Optional: Jersey Number Recognition (OCR) + Manual Mapping في البداية

**الجهاز:**
- Dell Precision 7740 مع NVIDIA RTX 3000 (CUDA متاح)
- هنستخدم 1080p @ 25-30 fps للحفاظ على السرعة

**المراحل المطلوبة:**
1. مرحلة 1: RTSP Stream + YOLO Detection + Tracking + Team Color
2. مرحلة 2: Court Mapping + Event Detection + Shots Analysis
3. مرحلة 3: Goalkeeper Specific Analysis + Notifications + Heatmaps
4. مرحلة 4: Full Tactical Analysis + Reports

الآن أريد منك:

أولاً: اكتب **Architecture كاملة** واضحة ومنظمة للمشروع (Pipeline، المكونات، كيفية الاتصال بينها).

ثانياً: خطة تطوير خطوة بخطوة (ما نبدأ بيه أولاً، والكود الأساسي لكل مرحلة).

ثالثاً: أفضل طريقة لقراءة الـ RTSP من الكاميرا + استخدام CUDA على RTX 3000.

رابعاً: نصائح مهمة لتقليل motion blur وتحسين دقة اكتشاف الكرة.

أخيرًا: لخص **كل** الرد السابق بأسلوب **Caveman** (بسيط جدًا، قصير، مثل رجل الكهف يتكلم: كلمات قليلة، مباشرة، بدون تعقيد).

ابدأ الآن.