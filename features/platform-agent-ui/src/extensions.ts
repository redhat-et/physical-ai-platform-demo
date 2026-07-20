const extensions = [
  {
    type: "app.navigation/href",
    properties: {
      id: "platform-agent",
      title: "Platform Agent",
      href: "/physicalAiStudio/platformAgent",
      section: "physical-ai-studio",
      path: "/physicalAiStudio/platformAgent/*",
      label: "Experimental",
      group: "1_platform-agent",
    },
  },
  {
    type: "app.route",
    properties: {
      path: "/physicalAiStudio/platformAgent/*",
      component: () => import("./PlatformAgent"),
    },
  },
];

export default extensions;
