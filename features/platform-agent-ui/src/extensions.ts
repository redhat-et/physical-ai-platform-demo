const extensions = [
  {
    type: "app.navigation/href",
    properties: {
      id: "platform-agent",
      label: "Platform Agent",
      href: "/physicalAiStudio/platformAgent",
      section: "physical-ai-studio",
      dataAttributes: {
        "data-testid": "platform-agent-nav",
      },
      description: "Experimental",
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
