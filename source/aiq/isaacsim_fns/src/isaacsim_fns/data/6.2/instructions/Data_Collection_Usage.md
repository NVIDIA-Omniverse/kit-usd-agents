# Data Collection & Usage

What data is being collected and how is it used?

Omniverse collects usage data when you install and start interacting with our software. The data we collect and how we use it are as follows:

> * Installation and configuration details such as version of your Operating System and Omniverse applications installed - This allows us to recognize usage trends and patterns
> * Hardware Details such as CPU, GPU, and display resolution - This allows us to optimize settings to provide best-in-class performance
> * Network configuration (Network speed) - This allows us to optimize to improve your experience
> * Product session and feature usage - This allows us to understand your user journey and product interaction to further enhance workflows
> * Error and crash logs - This allows us to improve performance & stability for troubleshooting and diagnostic purposes of our software

For more details, please see the NVIDIA Privacy Policy: <https://www.nvidia.com/en-us/about-nvidia/privacy-policy/>

---

How can I change my data collection settings when using Omniverse Kit containers?

NVIDIA Omniverse Kit Containers collect anonymous usage data to help improve software performance and aid diagnostic purposes. No personally identifiable information such as email addresses or names is collected.

To disable data collection for Kit Containers, set the following environment variable `OMNI_TELEMETRY_DISABLE_ANONYMOUS_DATA` to `1`.

---

How can I change my data collection settings when using the Omniverse Kit App Template?

The NVIDIA Omniverse Kit App Template collects anonymous usage data to help improve software performance and aid diagnostic purposes. No personally identifiable information such as email addresses or names is collected.

To disable data collection for the Kit App Template, follow these steps:

1. After creating an application with the `template new` tooling, open the `source/apps` directory
2. Locate the `.kit` file for the application you want to disable telemetry for
3. Within the `.kit` file locate:

   > ```python
   > [settings.telemetry]
   > # Anonymous Kit application usage telemetry
   > enableAnonymousData = true
   > ```
4. Change the `enableAnonymousData` variable to `false`

---

How can I change my data collection settings when using Omniverse Launcher version 1.85 & greater?

While data is necessary to provide services, you may opt out of data collection at any point in time. By default, we collect all usage and crash logs. To enable/disable data collection, follow these steps:

> 1. Open Omniverse Launcher
> 2. Click the **user icon** then **Settings**
> 3. Click **Data Collection**
> 4. **Enable/Disable** data collection based on your preference
> 5. Click **Save**, then click **Save** again on the Setting dialog

---

How can I change my Email Communication Preferences?

Visit <https://www.nvidia.com/en-us/about-nvidia/privacy-center/> and click **Manage My Email Preferences**.

---

How can I request the data NVIDIA Omniverse has collected? (Omniverse Launcher version 1.85 & greater.)

Visit <https://www.nvidia.com/en-us/about-nvidia/privacy-center/> and click **Request My Data**. Follow the request process and you will receive an email with the data we’ve collected.

---

How can I request deletion of the data NVIDIA Omniverse has collected? (Omniverse Launcher version 1.85 & greater.)

Visit <https://www.nvidia.com/en-us/about-nvidia/privacy-center/> and click **Delete My Data**. NVIDIA services will delete all your data and send you a confirmation email upon completion.