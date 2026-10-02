# کنترل دقیق‌تر مودها و مقایسهٔ مش‌ها

این بخش همراه با `--modal-audit` در خط لولهٔ قبلی اجرا می‌شود. تحلیل حلگر جدیدی را خودکار شروع نمی‌کند. ورودی‌های مدل، مش و تماس تغییر نکرده‌اند.

## چه چیزی اضافه شده است؟

* کران کمینه و بیشینهٔ سهم L/D/G در **تمام ترکیب‌های فضای هر خوشهٔ مقادیر ویژهٔ نزدیک**؛ میانگین بالا به‌تنهایی برای اعلام خلوص کافی نیست. برای برچسب پایدار، کمینهٔ سهم باید از آستانهٔ غالب بودن بیشتر باشد. سهم‌ها همچنان با نوع مبنا مشخص می‌شوند: هندسی یا مکانیکی. سهم هندسی، انرژی یا سهم بار کمانشی نیست.
* حساسیت این کران‌ها به تعریف پانل، کنترل افت رتبه و کنترل بیشترین خطای بازسازی در خوشهٔ مکانیکی. خوشهٔ آخر به علت نداشتن شکاف طیفی مشاهده‌شده در سمت بالا، برای پذیرش DSM باز می‌ماند.
* ذخیرهٔ تمام U1/U2/U3 گره‌ها با دقت محاسباتی در `modal_shapes.npz`؛ شکل کوچک‌شدهٔ نمودارها وارد مقایسه نمی‌شود.
* تطبیق فضاهای مود با زاویه‌های اصلی و تخصیص یک‌به‌یک؛ شمارهٔ مود، علامت و مقیاس بردار معیار تطبیق نیستند. ابعاد متفاوت خوشه‌ها، تطبیق ضعیف یا اختلاف زیاد مقدار ویژه رد می‌شود.
* علاوه بر امضای هندسه و مصالح، کلیدواژه‌های فیزیکی INP نیز مقایسه می‌شوند. این کنترل جای بازبینی دقیق نواحی تکیه‌گاه، تبدیل MPC و وضعیت فعال تماس را نمی‌گیرد.

## روش مقایسهٔ سطوح خمیده

طول وترهای مش ۲۰ و ۱۰ میلی‌متری روی دیوارهٔ خمیده متفاوت است. بنابراین پارامتر مقطع از **طول قوس پلی‌لاین ورودی مشترک** گرفته می‌شود. قرارگیری گره‌ها روی همان پلی‌لاین و یک‌به‌یک و صعودی بودن نگاشت کنترل می‌شود. گره خارج از هندسهٔ مرجع یا توپولوژی منشعب، بی‌صدا درون‌یابی نمی‌شود.

U هر دو مش روی این مقطع مرجع و ایستگاه‌های طولی مشترک نگاشت می‌شود. ضرب میدان‌های دوخطی با دو نقطهٔ گاوس روی اجتماع بازه‌های هر دو مش انتگرال‌گیری می‌شود. این انتگرال برای میدان **نگاشت‌شده** دقیق است؛ اختلاف تقریب هندسی سطوح پوسته‌ای جداگانه گزارش می‌شود. چرخش UR و همگرایی تنش/انرژی جزو معیار فعلی تطبیق شکل نیستند.

آستانه‌های پیش‌فرض: اختلاف مقدار ویژه ۵٪، کمینهٔ مربع کسینوس زاویهٔ اصلی ۰٫۹۵، پهنای نسبی خوشه ۰٫۰۰۱. این‌ها معیارهای عددی قابل تغییرند، نه حدود آیین‌نامه‌ای یا احتمال صحت. تغییر آستانه فقط برای افزایش تعداد نتیجهٔ پذیرفته‌شده توجیه ندارد. دو مش فقط یک مقایسه فراهم می‌کنند؛ برای همگرایی مجانبی، پالایش متوالی لازم است.

## اجرای پس‌پردازش موجود، بدون حل مجدد

در CMD، `RUN20` و `RUN10` را به پوشه‌های خروجی واقعی اشاره دهید. پوشهٔ خروجی گزارش باید جدید یا خالی باشد.

```bat
cd /d "D:\CFS-Column\New folder (6)"
set "CASE=A4784_t3_qm1_0_0_0_R2p5_lipS_lipLen60_M80_L3600_gap10_nb19_end25-25_row15"
set "RUN20=D:\CFS-Column\New folder (8)\%CASE%"
set "RUN10=D:\CFS-Column\New folder (9)\%CASE%"

abaqus python abaqus_dsm_modal_audit.py --run-dir "%RUN20%" --output-dir "%RUN20%\modal_validation_new"
abaqus python abaqus_dsm_modal_audit.py --run-dir "%RUN10%" --output-dir "%RUN10%\modal_validation_new"

abaqus python abaqus_modal_validation.py ^
  --mesh-a "%RUN20%\modal_validation_new\modal_shapes.npz" ^
  --mesh-b "%RUN10%\modal_validation_new\modal_shapes.npz" ^
  --output-dir "%RUN20%\modal_validation_new\mesh20_vs_10"
```

گزارش‌های همین بررسی از قبل در `modal_validation` ساخته شده‌اند؛ برای دیدن آن‌ها اجرای دوباره لازم نیست. فایل اصلی `eigenspace_validation.html` و نتیجهٔ مقایسه `mesh20_vs_10/mesh_comparison.json` است. `modal_explorer.html` شکل‌ها و درصدهای هر مود را نشان می‌دهد؛ ستون `eigenspace_stable_family` و فایل بازه‌ها مرجع تصمیم دربارهٔ پایداری خوشه‌اند.

برای پذیرش مستند DSM، علاوه بر مبنای معتبر و مرجع مستقل، گزینهٔ جدید `--shape-comparison` باید به گزارش تطبیق **همین دو ODB** اشاره کند؛ تطبیق هش فایل‌ها و مود نامزد هر خانواده کنترل می‌شود. فقط نزدیکی دو تنش یا یادداشت دستیِ همگرایی کافی نیست. نبود نامزد، نبود آن خانواده در طیف کامل را ثابت نمی‌کند.

## هستهٔ نیروپایه و حدود کاربرد آن

`ForceProjector` برای ماتریس‌های سازگار و مقید K، J و E این فضاها را جدا می‌کند:

```
L = ker(J.T)
GD = range(K^-1 J)
D = K^-1 J ker(E)
G = مکمل متعامد D در GD، با معیار K
```

K باید سختی الاستیک مثبت‌معینِ همان درجات آزادی باشد؛ J بارهای دیواره و E تعادل آن‌هاست. همه باید از **همان تبدیل قیود و تماس** عبور کرده باشند. برای فضای کامل، تعریف فیزیکی دیواره‌ها ضروری است؛ روی فضای کاهش‌یافته فقط تفکیک همان فضای کاهش‌یافته به دست می‌آید. پایهٔ مکانیکی از برچسب هندسی یا S/E یک مود ساخته نمی‌شود.

قالب NPZ جدید `metadata.format="force_based_KJE"` از قرارداد نگاشت گره/DOF و K موجود در README_dsm_modal_audit پیروی می‌کند و به جای L/D/G صریح، آرایه‌های J و E را می‌گیرد. علاوه بر فرادادهٔ قبلی، `wall_definition`، `equilibrium_definition`، `constraint_mapping_review` و `contact_state_review` ضروری‌اند. این مسیر legacy برای NPZ خارجیِ متراکم به ۵۰۰۰ درجهٔ آزادی مقید محدود است. **Stage A خودکار جدید از این محدودیت استفاده نمی‌کند**: برای مدل چهارقطعه‌ای، canonical finite-strip reference مستقل از مش Abaqus ساخته می‌شود و برای هر harmonic، `K0/J_GD/J_D` و basisهای L/D/G به‌صورت خودکار و bolt-independent تولید و cache می‌شوند. استفاده از یک زیرماتریس دلخواه K، حذف قیود یا تفسیر انرژی خمشی به‌عنوان انرژی موضعی همچنان مجاز نیست.

اسکریپت legacy `verify_force_projector_benchmark.py` فقط وقتی قابل استفاده است که یک MAT مستقل و مستند از CUFSM در اختیار باشد. repository فعلی **هیچ نتیجهٔ native ثبت‌شده‌ای را به‌عنوان pass ادعا نمی‌کند**؛ نبود فایل MAT/JSON خارجی نباید با تست synthetic جایگزین یا به‌عنوان تأیید CUFSM گزارش شود.

## مرجع و محدودیت علمی

* تعریف نیروپایهٔ Python از روی توابع اصلی CUFSM 5.70 در همین repository پیاده‌سازی شده است، اما **تأیید native MATLAB/CUFSM تا زمان اجرای benchmark خارجی زیر، pending است**. خود `SecAnal_fcFSM.m` نیز پشتیبانی‌نکردن از قیود کاربر را صریحاً بیان می‌کند؛ بنابراین اعمال مستقیم آن به ستون پیچ‌دار به‌جای reference classifier قابل قبول نیست.
* [معرفی رسمی CUFSM و روش‌های مقید](https://www.ce.jhu.edu/cufsm/about/)
* [مستندات تحلیل کمانش Abaqus](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEANLRefMap/simaanl-c-eigenbuckling.htm): وضعیت تماس در پایهٔ تحلیل کمانش ثابت می‌ماند.
* mode number و eigenvalue ابتدا از description فریم استخراج می‌شوند. `frameValue` فقط وقتی به‌عنوان مقدار دقیق‌تر eigenvalue پذیرفته می‌شود که با مقدار چاپ‌شده و resolution آن سازگار باشد؛ مقدار ناسازگار رد می‌شود. دقتی بیش از شواهد ذخیره‌شدهٔ ODB ادعا نمی‌شود.

گزارش CUFSM موجود برای این مقطع، Fcre را با منبع `Abaqus shell FE` ذخیره کرده است. آن مقدار مرجع مستقل برای تأیید خود Abaqus نیست. همچنین مرجع تک‌قطعه‌ای L/D و مدل اتصال بدون تماس را نباید بدون اثبات هم‌ارزی به‌عنوان خانواده‌های ستون کامل پذیرفت.


## بنچمارک مستقل classifier مکانیکی جدید با CUFSM/fcFSM 5.70

برای Stage A جدید، تست‌های synthetic فقط sanity check جبری هستند و **جای اجرای واقعی CUFSM را نمی‌گیرند**. مسیر مستقل اکنون end-to-end است و از دو فایل repository استفاده می‌کند:

- `benchmark_fcfsm_classifier_cufsm.m`: مرجع native را مستقیماً با توابع CUFSM 5.70 تولید می‌کند؛
- `verify_fcfsm_classifier_benchmark.py`: همان geometry/material/BC/harmonic را با implementation واقعی Stage A بازسازی و مقایسه می‌کند.

benchmark از یک **lipped open section پنج‌دیواره‌ای** استفاده می‌کند تا فضای Distortional تعادلی غیرصفر باشد. MATLAB با توابع واقعی `SecAnal_fcFSM`, `klocal`, `trans`, `assemble` و `elemprop`، ماتریس `K0`، basisهای L/D/G و سهم‌های یک probe مشترک را export می‌کند. Python سپس بدون فایل classifier دست‌ساز، basis را با `fcfsm_reference_basis.py` می‌سازد.

مقایسهٔ schema v2 هم‌زمان این موارد را کنترل می‌کند:

- سازگاری دقیق geometry، thickness، material، `S-S`، member length و harmonic number؛
- خطای نسبی Frobenius بین `K0_native` و `K0_Python`;
- اختلاف سهم K0-energy برای L/D/G؛
- principal-angle/subspace agreement برای basisهای L/D/G، مستقل از sign، scale و rotation داخلی basis.

### اجرای مرجع native در MATLAB/CUFSM

از root repository:

```matlab
cd('D:\CFS-Column\git_hub')
benchmark_fcfsm_classifier_cufsm
```

خروجی پیش‌فرض:

```text
D:\CFS-Column\git_hub\cufsm_fcfsm_reference.json
```

سپس در Command Prompt:

```bat
cd /d "D:\CFS-Column\git_hub"

python verify_fcfsm_classifier_benchmark.py ^
  --reference "cufsm_fcfsm_reference.json" ^
  --output "fcfsm_classifier_comparison.json"
```

در این حالت `--classifier` لازم نیست؛ ابزار Python خودش classifier-side fixture را با **کد Stage A فعلی** تولید می‌کند. گزینهٔ `--classifier` فقط برای بررسی یک fixture از پیش ساخته‌شده باقی مانده است.

پیش‌فرض acceptance عددی benchmark:

- family share difference ≤ 1 percentage point؛
- minimum cosine-squared ≥ 0.99 برای هر زیرفضای L/D/G؛
- relative Frobenius error of K0 ≤ `1e-8`.

این‌ها معیار numerical implementation validation هستند، نه حد آیین‌نامه‌ای و نه اثبات نهایی مدل چهارقطعه‌ای پیچ‌دار.

تا زمانی که دو دستور بالا روی MATLAB/CUFSM واقعی اجرا نشده و `external_status = PASSED` در `fcfsm_classifier_comparison.json` ثبت نشده باشد، وضعیت external benchmark باید **NOT EXECUTED / PENDING** تلقی شود. هیچ تست synthetic یا GitHub CI اجازه ندارد جای آن را بگیرد.

