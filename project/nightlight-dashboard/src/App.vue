<template>
  <div id="layout">
    <NavBar />
    <main class="main-content">
      <RouterView v-slot="{ Component, route }">
        <Transition name="page">
          <component :is="Component" :key="route.path" />
        </Transition>
      </RouterView>
    </main>
  </div>
</template>

<script setup>
import NavBar from '@/components/NavBar.vue'
</script>

<style>
#layout {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
}

.main-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  padding-top: var(--nav-h);
  position: relative;
  background:
    linear-gradient(180deg, rgba(3,13,26,0.15) 0%, rgba(3,13,26,0.55) 100%),
    url('/earth-night.jpg') center center / cover no-repeat fixed;
}

/* Page transition — smooth crossfade */
.page-enter-active {
  transition: opacity 0.4s cubic-bezier(0.16, 1, 0.3, 1), transform 0.4s cubic-bezier(0.16, 1, 0.3, 1);
}
.page-leave-active {
  transition: opacity 0.25s ease;
}
.page-enter-from {
  opacity: 0;
  transform: translateY(20px);
}
.page-leave-to {
  opacity: 0;
}
</style>
